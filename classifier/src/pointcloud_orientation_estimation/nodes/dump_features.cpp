/*
Copyright (c) 2014-2015, Timm Linder, Social Robotics Lab, University of Freiburg
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <string>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <chrono>
#include <ctime>

#include <omp.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/ml/ml.hpp>

#include "../volume.h"
#include "../volume_visualizer.h"
#include "../tessellation_generator.h"
#include "../features.h"
#include "../classifier.h"

#include "../3rd_party/cnpy/cnpy.h"

using namespace std;

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// THIS ENTIRE BLOCK IS JUST TO DUMP USEFUL STATISTICS ABOUT THE RESULTS OF THE TRAINING PROCESS

struct CloudInfo {
    pcl::PointXYZ pose, velocity;
    float thetaDeg; // person orientation. 180° = looking INTO camrea
    float sensorDistance;
    float phi; // angle between optical axis of camera and person
    size_t numPoints;
};

// For looking up which column in a sample's feature vector belongs to which feature of which volume of which tessellation.
struct FeatureVectorLookupEntry {
    size_t tessellationIndex;
    size_t volumeInTessellationIndex;
    size_t overallVolumeIndex;
    size_t featureIndex;
};

struct HistogramEntry {
    size_t numberOfTimesUsed;
    float accumulatedQuality;
};

// To detect which features are being used the most often
struct FeatureHistogramEntry : public HistogramEntry {
    size_t featureIndex;
};

// To detect which volumes (voxels) are being used the most often
struct VolumeHistogramEntry  : public HistogramEntry {
    size_t overallVolumeIndex;
    size_t tessellationIndex;
    size_t volumeInTessellationIndex;
};

struct HistogramEntryComparatorByNumberOfTimesUsed {
    bool operator() (const HistogramEntry& lhs, const HistogramEntry& rhs) const {
       return lhs.numberOfTimesUsed > rhs.numberOfTimesUsed;
    }
};

struct HistogramEntryComparatorByAccumulatedQuality {
    bool operator() (const HistogramEntry& lhs, const HistogramEntry& rhs) const {
       return lhs.accumulatedQuality > rhs.accumulatedQuality;
    }
};

// For sorting splits (decision stumps) of Adaboost classifier
struct SplitComparatorByQuality {
    bool operator() ( CvDTreeSplit* const lhs, CvDTreeSplit* const rhs) const {
       return lhs->quality > rhs->quality;
    }
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Global variables
std::vector<string> g_featureNames;
Volume g_parentVolume;
size_t g_numPositiveSamples, g_numNegativeSamples, g_minPoints;
bool g_dumpFeatures;
std::string g_category, g_numpyExportFolder;
double g_scaleZto, g_cropZmin, g_cropZmax;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool loadCloudInfo(std::string& poseFilename, CloudInfo& cloudInfo)
{
    std::ifstream poseFile(poseFilename.c_str());
    if(poseFile.fail()) {
        return false;
    }

    std::string fieldName, NaNString;

    float NaN = std::numeric_limits<float>::quiet_NaN();
    cloudInfo.numPoints = 0;
    cloudInfo.pose.x = NaN; cloudInfo.pose.y = NaN; cloudInfo.pose.z = 0;
    cloudInfo.velocity.x = NaN; cloudInfo.velocity.y = NaN; cloudInfo.velocity.z = 0;
    cloudInfo.thetaDeg = NaN; cloudInfo.sensorDistance = NaN;

    // The fail() / clear() mechanism is needed because istream cannot handle NaN values
    poseFile >> fieldName >> cloudInfo.pose.x;            if(poseFile.fail()) { poseFile.clear(); poseFile >> NaNString; }  ROS_ASSERT(fieldName == "position_x");
    poseFile >> fieldName >> cloudInfo.pose.y;            if(poseFile.fail()) { poseFile.clear(); poseFile >> NaNString; }  ROS_ASSERT(fieldName == "position_y");
    poseFile >> fieldName >> cloudInfo.velocity.x;        if(poseFile.fail()) { poseFile.clear(); poseFile >> NaNString; }  ROS_ASSERT(fieldName == "velocity_x");
    poseFile >> fieldName >> cloudInfo.velocity.y;        if(poseFile.fail()) { poseFile.clear(); poseFile >> NaNString; }  ROS_ASSERT(fieldName == "velocity_y");
    poseFile >> fieldName >> cloudInfo.thetaDeg;          if(poseFile.fail()) { poseFile.clear(); poseFile >> NaNString; }  ROS_ASSERT(fieldName == "theta_deg");
    poseFile >> fieldName >> cloudInfo.sensorDistance;    if(poseFile.fail()) { poseFile.clear(); poseFile >> NaNString; }  ROS_ASSERT(fieldName == "sensor_distance");

    cloudInfo.phi = atan2(cloudInfo.pose.y, cloudInfo.pose.x) * 180.0 / M_PI;
    return true;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void calculateFeaturesOnDataset(std::list<Tessellation>& tessellations,
    std::vector<FeatureVectorLookupEntry>& featureVectorLookup, std::vector<Volume>& overallVolumeLookup, std::vector<std::string>& cloudFilenames,
    std::vector<CloudInfo>& cloudInfos, std::string trainfile)
{
    PointCloud::Ptr personCloud(new PointCloud);

    // Count total number of volumes (voxels) over all tessellations
    size_t overallVolumeCount = 0;
    foreach(Tessellation tessellation, tessellations) {
        foreach(Volume volume, tessellation.getVolumes()) {
            overallVolumeCount++;
        }
    }

    // Count number of features
    FeatureCalculator featureCalculator;
    g_featureNames = featureCalculator.getFeatureNames();
    const size_t featureCount = g_featureNames.size();
    const size_t featureVectorSize = overallVolumeCount * featureCount;

    ROS_INFO_STREAM_ONCE("Total count of tessellation volumes (voxels):  " << overallVolumeCount);
    ROS_INFO_STREAM_ONCE("Number of different features: " << featureCount);
    ROS_INFO_STREAM_ONCE("--> Each person cloud will have a feature vector of dimension " << featureVectorSize);

    //
    // Load training data set and calculate features
    //
    //std::stringstream trainFilename(trainfile);
    //trainFilename << ros::package::getPath(ROS_PACKAGE_NAME) << "/data/" << g_category << "/fold" << std::setw(3) << std::setfill('0') << fold << "/" << (useValidationSet ? "val" : "train") << ".txt";
    //trainFilename
    std::ifstream listFile(trainfile.c_str());

    string cloudFilename;
    float label;
    size_t numClouds = 0;
    set<int> labelSet;

    // Skip comments at beginning of file
    std::string commentString;
    const size_t numCommentLinesAtBeginning = 0;
    for(size_t i = 0; i < numCommentLinesAtBeginning; i++) std::getline(listFile, commentString);

    // See how many cloud files we have
    while (listFile >> cloudFilename >> label)
    {
        numClouds++;
    }

    // Go back to start of list file
    listFile.clear();
    listFile.seekg(0, std::ios::beg);
    for(size_t i = 0; i < numCommentLinesAtBeginning; i++) std::getline(listFile, commentString);

    // Clear other fields
    g_numNegativeSamples = g_numPositiveSamples = 0;
    cloudFilenames.clear();
    cloudInfos.clear();

    // Create map for looking up which column in the feature vector belongs to which tessellation + voxel + feature
    overallVolumeLookup.clear();
    overallVolumeLookup.reserve(overallVolumeCount);
    featureVectorLookup.clear();
    featureVectorLookup.reserve(featureVectorSize);
    bool featureVectorLookupInitialized = false;


    ROS_INFO_STREAM("Loading " << numClouds << " clouds to calculate features...");

    // Now start calculating features
    size_t numSkippedLowQualityClouds = 0;
    ros::WallRate rate(10);
    int goodCloudCounter = 0, overallCloudCounter = 0, skippedCloudCounter = 0;
    auto start = std::chrono::system_clock::now();
    while (listFile >> cloudFilename >> label)
    {

    	auto end = std::chrono::system_clock::now();
    	        auto elapsed = end - start;
    	        auto hours = chrono::duration_cast<chrono::hours>(elapsed);
    	        auto minutes = chrono::duration_cast<chrono::minutes>(elapsed);
    	        auto seconds = chrono::duration_cast<chrono::seconds>(elapsed);
    	        auto estimation = (end - start)*((float(numClouds-skippedCloudCounter) - float(goodCloudCounter-skippedCloudCounter)) / float(goodCloudCounter-skippedCloudCounter));
    	        auto estihours = chrono::duration_cast<chrono::hours>(estimation);
    	        auto estiminutes = chrono::duration_cast<chrono::minutes>(estimation);
    	        auto estiseconds = chrono::duration_cast<chrono::seconds>(estimation);
    	        std::cout << "File : " << goodCloudCounter << "/" << numClouds << " = " << float(goodCloudCounter) / float(numClouds) * 100.0 << "%"
    	        		  << " elapsed time : " << hours.count() << "h" << minutes.count()%60 << "m" << seconds.count()%60 << "s"
    	        		  << " estimated time till finish : " << estihours.count() << "h" << estiminutes.count()%60 << "m" << estiseconds.count()%60 << "s" << "\r";
    	boost::filesystem::path stopiIt("stopit");
		if(boost::filesystem::exists(stopiIt))
		{
			std::cout << "some high entity signaled me to stop ... so i do" << std::endl;
			break;
		}


    	cv::Mat features;
        features.create(1, featureVectorSize, CV_32FC1);
        features.setTo(cv::Scalar(std::numeric_limits<double>::quiet_NaN()));

        cv::Mat missingDataMask;
        missingDataMask.create(1, featureVectorSize, CV_8UC1);
        missingDataMask.setTo(cv::Scalar(1));

        /*if(overallCloudCounter % (numClouds / 10) == 0) {
            ROS_INFO("%d %% of feature computations done...", int(overallCloudCounter / (float)numClouds * 100.0f + 0.5f));
        }*/

        // Load pose file
        std::string poseFilename = cloudFilename;
        boost::replace_all(poseFilename, "_cloud.pcd", "_pose.txt");

        CloudInfo cloudInfo;
        if(!loadCloudInfo(poseFilename, cloudInfo)) {
            ROS_FATAL("Couldn't read pose file %s\n", poseFilename.c_str());
            continue;
        }
        
        // Load PCD file
        if(pcl::io::loadPCDFile<PointType>(cloudFilename, *personCloud) == -1)
        {
            ROS_FATAL("Couldn't read file %s\n", cloudFilename.c_str());
            continue;
        }

        // Number of points is the last meta-data item we need to decide about goodness of cloud
        cloudInfo.numPoints = personCloud->points.size();

        // Everything good! Now store meta data and name of input cloud (for dump of results later on)
        cloudInfos.push_back(cloudInfo);
        cloudFilenames.push_back(cloudFilename);
        
        if(label > 0) g_numPositiveSamples++;
        else g_numNegativeSamples++;

        // Scale point cloud in z direction (height)
        scaleAndCropCloudToTargetSize(personCloud, g_scaleZto, g_cropZmin, g_cropZmax);

        // Calculate features
        std::vector<double> fullFeatureVectorForCloud;

        size_t featureColumn = 0, t = 0, overallVolumeIndex = 0;
        foreach(Tessellation& tessellation, tessellations)  // for each tessellation...
        {
            #pragma omp parallel for schedule(dynamic) ordered
            for(size_t v = 0; v < tessellation.getVolumes().size(); v++)  // for each volume in that tessellation...
            {
                // Get points inside volume
                std::vector<int> indicesInsideVolume;
                const Volume& volume = tessellation.getVolumes()[v];
                volume.getPointsInsideVolume(*personCloud, PointCloud::Ptr(), &indicesInsideVolume);
                //std::cout << indicesInsideVolume.size() << personCloud->points.size() << std::endl;
                // Calculate features (if sufficient points inside volume)
                std::vector<double> volumeFeatureVector;
                const size_t MIN_POINT_COUNT = g_minPoints;
                if(indicesInsideVolume.size() >= MIN_POINT_COUNT) {
                    featureCalculator.calculateFeatures(g_parentVolume, *personCloud, indicesInsideVolume,
                        featureCalculator.maskAllFeaturesActive(), volumeFeatureVector); 
                }
                else volumeFeatureVector = std::vector<double>(featureCount, std::numeric_limits<double>::quiet_NaN());

                // Copy feature values into right spot of sample's overall feature vector
                ROS_ASSERT(volumeFeatureVector.size() == featureCount);
                    
                #pragma omp ordered
                #pragma omp critical
                {
                    for(size_t f = 0; f < volumeFeatureVector.size(); f++)  // for each feature...
                    {
                        features.at<float>(0, featureColumn) = volumeFeatureVector[f];
                        missingDataMask.at<unsigned char>(0, featureColumn) = !std::isfinite(volumeFeatureVector[f]) ? 1 : 0;

                        if(!featureVectorLookupInitialized) {
                            FeatureVectorLookupEntry entry;
                            entry.tessellationIndex = t;
                            entry.volumeInTessellationIndex = v;
                            entry.overallVolumeIndex = overallVolumeIndex;
                            entry.featureIndex = f;
                            featureVectorLookup.push_back(entry);
                        }
                        featureColumn++;
                    }

                    if(!featureVectorLookupInitialized) {
                        overallVolumeLookup.push_back(volume);
                    }

                    overallVolumeIndex++;
                }
            }
            t++;
        }

        bool serializeOpenCv = true;
        if(serializeOpenCv == true)
        {
            boost::filesystem::path p(cloudFilename);
            boost::filesystem::path dir = p.parent_path().parent_path().parent_path();
            dir += "/features/";
            if(!boost::filesystem::exists(dir))
            {
                    boost::filesystem::create_directory(dir);
            }
            std::string featureFileName = p.filename().string();
            boost::replace_all(featureFileName, "_cloud.pcd", "_features.xml");
            goodCloudCounter++;
            dir += featureFileName;
            //dir = "/media/mrtimmer/638406EC514C363F/NIKROrientationDataset/data_set/p35/features/p35-jacke-Kinect1-330-Take1_features.xml";
            /*if(boost::filesystem::exists(dir))
            {
                    skippedCloudCounter++;
                    continue;
            }*/
            cv::FileStorage file(dir.string().c_str(), cv::FileStorage::WRITE );
            file << "fileName" << cloudFilename.c_str();
            file << "index" << boost::lexical_cast<string>(goodCloudCounter);
            file << "numPoints" << static_cast<int>(cloudInfo.numPoints);
            file << "thetaDeg" << cloudInfo.thetaDeg;
            file << "features" << features;
            file << "missing" << missingDataMask;
            file.release();
        }
        
        bool serializeNumpy = false;
        if(serializeNumpy == true)
        {
            boost::filesystem::path p(cloudFilename);
            boost::filesystem::path NpFiledir = p.parent_path().parent_path().parent_path();
            NpFiledir += "/Numpyfeatures/";
            if(!boost::filesystem::exists(NpFiledir))
            {
                    boost::filesystem::create_directory(NpFiledir);
            }
            
            std::string NumpyfeatureFileName = p.filename().string();
            boost::replace_all(NumpyfeatureFileName, "_cloud.pcd", "");
            NpFiledir += NumpyfeatureFileName;
            const unsigned int featuresShape[] = { features.rows, features.cols };
            cnpy::npy_save( NpFiledir.string() + "_features.npy", (float*)features.data, featuresShape, 2, "w" );

            const unsigned int missingShape[] = { missingDataMask.rows, missingDataMask.cols };
            cnpy::npy_save( NpFiledir.string() + "_missing.npy", (unsigned char*)missingDataMask.data, missingShape, 2, "w" );
        }
        goodCloudCounter++;
        featureVectorLookupInitialized = true;
        ROS_ASSERT(featureVectorLookup.size() == featureVectorSize);
        ROS_ASSERT(overallVolumeLookup.size() == overallVolumeCount);

        // Prepare for next sample


    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void generatePermutationsOfAspectRatios(std::vector<pcl::PointXYZ>& voxelAspectRatios)
{
    std::vector<pcl::PointXYZ> newAspectRatios;

    foreach(pcl::PointXYZ aspectRatio, voxelAspectRatios) {
        newAspectRatios.push_back( pcl::PointXYZ(aspectRatio.x, aspectRatio.z, aspectRatio.y) );
        newAspectRatios.push_back( pcl::PointXYZ(aspectRatio.z, aspectRatio.x, aspectRatio.y) );
        newAspectRatios.push_back( pcl::PointXYZ(aspectRatio.z, aspectRatio.y, aspectRatio.x) );
        newAspectRatios.push_back( pcl::PointXYZ(aspectRatio.y, aspectRatio.x, aspectRatio.z) );
        newAspectRatios.push_back( pcl::PointXYZ(aspectRatio.y, aspectRatio.z, aspectRatio.x) );       
    }

    voxelAspectRatios.insert(voxelAspectRatios.end(), newAspectRatios.begin(), newAspectRatios.end());
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    ros::init(argc, argv, "top_down_classifier_training");
    ros::NodeHandle nodeHandle("");
    ros::NodeHandle privateHandle("~");
    std::string trainfile;

    privateHandle.param<bool>("dump_features", g_dumpFeatures, false);

    //privateHandle.param<string>("npy_export_folder", g_numpyExportFolder, "");
    //privateHandle.param<string>("category", g_category, "");
    privateHandle.param<string>("trainfile", trainfile, "");
    //ROS_ASSERT_MSG(!g_category.empty(), "_category must be specified (e.g. 'gender')");


    g_minPoints = 4; 

    omp_set_num_threads(5);

    privateHandle.param<double>("scale_z_to", g_scaleZto, 0.0);
    privateHandle.param<double>("crop_z_min", g_cropZmin, -std::numeric_limits<double>::infinity());
    privateHandle.param<double>("crop_z_max", g_cropZmax, +std::numeric_limits<double>::infinity());
    ROS_INFO("Input cloud scaling factor in z direction is %.3f", g_scaleZto > 0 ? g_scaleZto : 1.0f);
    ROS_INFO("Cropping input clouds in z direction between %.3f and %.3f", g_cropZmin, g_cropZmax);

    double parentVolumeWidth, parentVolumeHeight, parentVolumeZOffset;
    privateHandle.param<double>("parent_volume_width", parentVolumeWidth, 0.6);
    privateHandle.param<double>("parent_volume_height", parentVolumeHeight, 1.8);
    privateHandle.param<double>("parent_volume_z_offset", parentVolumeZOffset, 0);
    
    const double parentVolumeHalfWidth = 0.5 * parentVolumeWidth;

    pcl::PointXYZ minCoords(-parentVolumeHalfWidth, -parentVolumeHalfWidth, parentVolumeZOffset),
                  maxCoords(+parentVolumeHalfWidth, +parentVolumeHalfWidth, parentVolumeZOffset + parentVolumeHeight);

    g_parentVolume = Volume(minCoords, maxCoords); // = Volume::fromCloudBBox(*personCloud);

    pcl::PointXYZ parentVolumeSize = g_parentVolume.getSize();
    ROS_INFO("Parent volume has size %.2g %.2g %.2g", parentVolumeSize.x, parentVolumeSize.y, parentVolumeSize.z);

    //
    // Initialize tessellation generator
    //

    double minVoxelSize, regularTessellationSize;
    bool overlapEnabled, regularTessellationOnly;

    privateHandle.param<bool>("overlap", overlapEnabled, true);
    privateHandle.param<bool>("regular_tessellation_only", regularTessellationOnly, false);
    privateHandle.param<double>("regular_tessellation_size", regularTessellationSize, 0.1);
    privateHandle.param<double>("min_voxel_size", minVoxelSize, 0.1);
    
    
    std::vector<pcl::PointXYZ> voxelAspectRatios; 

    voxelAspectRatios.push_back( pcl::PointXYZ(1, 1, 1) );
    
    if(!regularTessellationOnly) {
        voxelAspectRatios.push_back( pcl::PointXYZ(1.0, 1.0, 2.5) );
        voxelAspectRatios.push_back( pcl::PointXYZ(1.0, 1.0, 5.0) );
        voxelAspectRatios.push_back( pcl::PointXYZ(1.0, 1.0, 4.0) );
        voxelAspectRatios.push_back( pcl::PointXYZ(1.0, 1.0, 6.0) );
        voxelAspectRatios.push_back( pcl::PointXYZ(1.0, 1.0, 8.0) );
        voxelAspectRatios.push_back( pcl::PointXYZ(1.0, 1.0, 10.0) );

        voxelAspectRatios.push_back( pcl::PointXYZ(0.1, 0.1, 1.8) );
        voxelAspectRatios.push_back( pcl::PointXYZ(0.2, 0.2, 1.8) );
        voxelAspectRatios.push_back( pcl::PointXYZ(0.3, 0.3, 1.8) );

        voxelAspectRatios.push_back( pcl::PointXYZ(1.0, 1.0, 1.25) );

        voxelAspectRatios.push_back( pcl::PointXYZ(2, 2, 2) );
        voxelAspectRatios.push_back( pcl::PointXYZ(3, 3, 3) );
        voxelAspectRatios.push_back( pcl::PointXYZ(4, 4, 4) );

        voxelAspectRatios.push_back( pcl::PointXYZ(1, 1, 2) );
        voxelAspectRatios.push_back( pcl::PointXYZ(1, 1, 3) );
        voxelAspectRatios.push_back( pcl::PointXYZ(2, 2, 3) );
        voxelAspectRatios.push_back( pcl::PointXYZ(2, 3, 3) ); // new
        voxelAspectRatios.push_back( pcl::PointXYZ(4, 4, 3) );
        voxelAspectRatios.push_back( pcl::PointXYZ(4, 4, 2) );
        voxelAspectRatios.push_back( pcl::PointXYZ(1, 4, 4) );
    }

    generatePermutationsOfAspectRatios(voxelAspectRatios);


    std::vector<float> voxelSizeIncrements;
    if(!regularTessellationOnly) {
        voxelSizeIncrements.push_back(0.1);
        voxelSizeIncrements.push_back(0.2);
        voxelSizeIncrements.push_back(0.3);
        voxelSizeIncrements.push_back(0.4);
        voxelSizeIncrements.push_back(0.5);
        voxelSizeIncrements.push_back(0.6);
        voxelSizeIncrements.push_back(0.7);
        voxelSizeIncrements.push_back(0.8);
        voxelSizeIncrements.push_back(0.9);
        voxelSizeIncrements.push_back(1.0);
        voxelSizeIncrements.push_back(1.1);
    }
    else {
        voxelSizeIncrements.push_back(regularTessellationSize);
    } 

    TessellationGenerator tessellationGenerator(g_parentVolume, voxelAspectRatios, voxelSizeIncrements, minVoxelSize, overlapEnabled);


    //
    // Generate tessellations
    //

    ROS_INFO_STREAM("Beginning to generate tessellations (overlap " << (overlapEnabled ? "enabled" : "disabled") << ")..." );
    std::list<Tessellation> tessellations;
    tessellationGenerator.generateTessellations(tessellations);
    ROS_INFO_STREAM("Finished generating tessellations! Got " << tessellations.size() << " in total!");

	ROS_INFO_STREAM("");
	ROS_INFO_STREAM("### STARTING TO DUMP FEATURES ");

	//
	// Calculate features on training set
	//

	cv::Mat labels, features, sampleIdx, missingDataMask;
	std::vector<std::string> cloudFilenames;
	std::vector<CloudInfo> cloudInfos;
    std::vector<FeatureVectorLookupEntry> featureVectorLookup;
    std::vector<Volume> overallVolumeLookup;
	calculateFeaturesOnDataset(tessellations, featureVectorLookup, overallVolumeLookup, cloudFilenames, cloudInfos,trainfile);

	ROS_INFO_STREAM("### DONE DUMPING FEATURES ");

    return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
