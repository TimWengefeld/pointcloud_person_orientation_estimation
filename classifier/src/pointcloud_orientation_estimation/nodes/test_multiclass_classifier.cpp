/*
* Software License Agreement (BSD License)
*
*  Copyright (c) 2014-2015, Timm Linder, Social Robotics Lab, University of Freiburg
*  All rights reserved.
*  
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions are met:
*  
*  * Redistributions of source code must retain the above copyright notice, this
*    list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*  * Neither the name of the copyright holder nor the names of its contributors
*    may be used to endorse or promote products derived from this software
*    without specific prior written permission.
*  
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
*  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
*  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
*  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
*  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
*  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
*  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
*  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
*  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

#include <omp.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/ml/ml.hpp>

#include "../volume.h"
#include "../volume_visualizer.h"
#include "../classifier.h"
#include "../features.h"

using namespace std;

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <chrono>
#define foreach BOOST_FOREACH


// Global variables
ros::Publisher g_pointCloudPublisher;
std::vector<boost::shared_ptr<TopDownClassifier> > g_multiclass_classifier;
std::string outputFileName;
bool use_clouds;

struct CloudInfo {
    pcl::PointXYZ pose, velocity;
    float thetaDeg; // person orientation. 180° = looking INTO camrea
    float sensorDistance;
    float phi; // angle between optical axis of camera and person
    size_t numPoints;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool loadCloudInfo(std::string& poseFilename, CloudInfo& cloudInfo)
{
	std::cout <<"Loading " << poseFilename << std::endl;
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
	/*cloudInfo.numPoints = 0;
    cloudInfo.pose.x = 2; cloudInfo.pose.y = 0; cloudInfo.pose.z = 0;
    cloudInfo.velocity.x = 0; cloudInfo.velocity.y = 0; cloudInfo.velocity.z = 0;
    cloudInfo.thetaDeg = 0; cloudInfo.sensorDistance = 2;*/
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
double testClassifier(const std::string& listFilename)
{
    std::ifstream listFile(listFilename.c_str());

    string cloudFilename;
    float groundtruthLabel;
    size_t numCloudsTotal = 0, numCloudsCorrectlyClassified = 0;
    
    // Skip comments at beginning of file
    std::string commentString;
    const size_t numCommentLinesAtBeginning = 0;

    // Process cloud by cloud individually
    ros::WallTime startTime = ros::WallTime::now();
    ros::WallDuration accumulatedDurationWithoutIO(0);
    std::vector<int> tp = std::vector<int>(g_multiclass_classifier.size(),0);
    std::vector<int> fp = std::vector<int>(g_multiclass_classifier.size(),0);
    std::vector<int> tn = std::vector<int>(g_multiclass_classifier.size(),0);
    std::vector<int> fn = std::vector<int>(g_multiclass_classifier.size(),0);
    std::vector<int> P = std::vector<int>(g_multiclass_classifier.size(),0);
    std::vector<int> N = std::vector<int>(g_multiclass_classifier.size(),0);
        
    boost::filesystem::path outputFilePath(outputFileName);
    if(!boost::filesystem::exists(outputFilePath.parent_path()))
    {
        boost::filesystem::create_directory(outputFilePath.parent_path());
    }
    ofstream csvFile(outputFileName);
    std::cout << "write results file to " << outputFileName << std::endl;
    csvFile << "classifier,mode,identifier,position_x,position_y,velocity_x,velocity_y,orientation,phi,num_points";
    for(int i = 0; i < g_multiclass_classifier.size();i++)
    {
            csvFile << ",score_" + g_multiclass_classifier[i]->getCategory();
    }
    csvFile << ",prediction";
    csvFile << std::endl;
        
    int readCounter = 0;
    ros::WallTime startTimeWithoutIO = ros::WallTime::now();
    std::chrono::duration<double, milli> duration(0);
    std::chrono::duration<double, milli> durationFeatures(0);
    std::chrono::duration<double, milli> durationClassify(0);
    while (listFile >> cloudFilename >> groundtruthLabel)
    {
    	std::cout << "blubb" << std::endl;
        // Load point cloud from disk
    	auto startLoad = std::chrono::system_clock::now();
        PointCloud::Ptr personCloud(new PointCloud);
        string filename = cloudFilename;
        std::cout <<"Loading " << filename << std::endl;

        if(pcl::io::loadPCDFile<PointType>(filename, *personCloud) == -1)
        {
            ROS_ERROR("Couldn't read file %s\n", filename.c_str());
            return (-1);
        }
        numCloudsTotal++;
        auto endLoad = std::chrono::system_clock::now();
        CloudInfo cloudInfo;
        std::string poseFilename = cloudFilename;
        boost::replace_all(poseFilename, "_cloud.pcd", "_pose.txt");
        loadCloudInfo(poseFilename, cloudInfo);
        
        double sumOfVotes;
        class_label label;

        cv::Mat feat;
        cv::Mat missing;
        if(!use_clouds)
        {
                boost::filesystem::path p(cloudFilename);
                boost::filesystem::path dir = p.parent_path().parent_path().parent_path();
                dir += "/features/";

                std::string featureFileName = p.filename().string();
                boost::replace_all(featureFileName, "_cloud.pcd", "_features.xml");

                dir += featureFileName;

                if(!boost::filesystem::exists(dir))
                {
                        std::cout << "couldn't load file " << dir.c_str() << " ABORT!!!!";
                        return 0;
                }

                std::cout << "Loaded " << readCounter << " Files" << std::endl;

                cv::FileStorage file( dir.c_str(), cv::FileStorage::READ );

                file["features"] >> feat;
                file["missing"] >> missing;
                file["fileName"] >> cloudFilename;
                file.release();
        }
        
        csvFile << "adaboost,test,";
        csvFile << cloudFilename << ",";
        csvFile << cloudInfo.pose.x << "," << cloudInfo.pose.y << "," << cloudInfo.velocity.x << "," << cloudInfo.velocity.y << "," << cloudInfo.thetaDeg << ",";
        csvFile << cloudInfo.phi << "," << cloudInfo.numPoints;
        float maxPrediction = -std::numeric_limits<float>::infinity();
        int maxLabel = 0;
        std::cout << "blubb" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        scaleAndCropCloudToTargetSize(personCloud, g_multiclass_classifier[0]->m_scaleZto,  g_multiclass_classifier[0]->m_cropZmin,  g_multiclass_classifier[0]->m_cropZmax);
        for(int i = 0; i < g_multiclass_classifier.size();i++)
        {
                if(use_clouds)
                {
                        cv::Mat featureVector, missingDataMask;
                        auto startFeatures = std::chrono::high_resolution_clock::now();
                        g_multiclass_classifier[i]->calculateFeatures(*personCloud,featureVector, missingDataMask);
                        auto endFeatures = std::chrono::high_resolution_clock::now();
                        auto startClassify = std::chrono::high_resolution_clock::now();
                        label = g_multiclass_classifier[i]->classifyFeatureVector(featureVector,missingDataMask,&sumOfVotes); // invoke classifier
                        auto endClassify = std::chrono::high_resolution_clock::now();
                        durationFeatures += endFeatures-startFeatures;
                        durationClassify += endClassify-startClassify;
                }
                
                else
                {
                        label = g_multiclass_classifier[i]->classifyFeatureVector(feat,missing,&sumOfVotes); // invoke classifier
                        std::cout << "Use Feature Vector" << std::endl;
                }
                
                std::string identifier = boost::filesystem::path(cloudFilename).stem().string();
                boost::replace_all(identifier, "_cloud", "");

                csvFile << "," << sumOfVotes;
                if(sumOfVotes > maxPrediction)
                {
                        maxPrediction = sumOfVotes;
                        maxLabel = i;
                }
        }
        auto end = std::chrono::high_resolution_clock::now();
        duration += end-start;
        std::cout << "blubb2" << std::endl;
        std::cout << "###################" << std::endl;
        std::cout << "Needed " << duration.count() << " milliseconds to estimate " << readCounter+1 << " clouds" << std::endl;
        std::cout << "This is on average " << duration.count()/(readCounter+1) << " milliseconds per cloud" << std::endl;

        std::cout << "Needed " << durationFeatures.count() << " milliseconds to calc features for " << readCounter+1 << " clouds" << std::endl;
        std::cout << "This is on average " << durationFeatures.count()/(readCounter+1) << " milliseconds per cloud" << std::endl;

        std::cout << "Needed " << durationClassify.count() << " milliseconds to classify features for " << readCounter+1 << " clouds" << std::endl;
        std::cout << "This is on average " << durationClassify.count()/(readCounter+1) << " milliseconds per cloud" << std::endl;
        std::cout << "###################" << std::endl;
        csvFile << "," << g_multiclass_classifier[maxLabel]->getCategory().c_str();
        csvFile << std::endl;

        readCounter++;
	}

    // Return accuracy on test set
    return (double) numCloudsCorrectlyClassified / numCloudsTotal;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    ros::init(argc, argv, "test_top_down_classifier");
    ros::NodeHandle nodeHandle("");
    ros::NodeHandle privateHandle("~");

    //
    // Parse arguments
    //

    bool showBestTessellation; std::string listFilename, modelFilename; int numThreads, numberOfVolumesToShow;

    privateHandle.param<bool>("show_best_tessellation", showBestTessellation, true);
    privateHandle.param<int>("num_volumes_to_show", numberOfVolumesToShow, 50);
    privateHandle.param<std::string>("list_file", listFilename, "");
    privateHandle.param<std::string>("model", modelFilename, "");
    privateHandle.param<int>("num_threads", numThreads, 5);
    privateHandle.param<bool>("use_clouds", use_clouds, false);
    privateHandle.param<std::string>("outputFileName", outputFileName, "evalfile.csv");

    omp_set_num_threads(8);
    ROS_INFO_STREAM("Using " << numThreads << " parallel threads for feature computations!");
    
    // Create point cloud publisher
    g_pointCloudPublisher = nodeHandle.advertise<sensor_msgs::PointCloud2>("cloud", 1, true);
    
    int mNumClassifiersToTest;
    std::vector<std::string> mClassifierPaths;
    cv::FileStorage fileStorage(modelFilename.c_str(), cv::FileStorage::READ);
    boost::filesystem::path modelBasePath(modelFilename.c_str());
    fileStorage["numClassifier"] >> mNumClassifiersToTest;
    for(int i = 0; i < mNumClassifiersToTest; i++)
    {
            std::string identifier = "classifier_" + std::to_string(i);
            std::string path;
            fileStorage[identifier] >> path;
            mClassifierPaths.emplace_back(modelBasePath.parent_path().string() + "/" + path);
            std::cout << "Load classifier : " << path << std::endl;
    }
    fileStorage.release();

    //
    // Load classifier
    //
    if(modelFilename.empty())
    {
        ROS_ERROR_STREAM("The _model argument was not specified; this is required, and must point to a YAML file containing the learned classifier.");
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < mClassifierPaths.size(); i++)
    {
        g_multiclass_classifier.push_back(boost::shared_ptr<TopDownClassifier>(new TopDownClassifier));
        std::cout << "load model from : " << mClassifierPaths[i] << std::endl;
        g_multiclass_classifier.back()->init(mClassifierPaths[i]);
    }

    //
    // Test classifier
    //

    if(!listFilename.empty())
    {
        // Test classifier on provided list file (each line contains a cloud filename + label
        // separated by space or tabulator; first 2 lines are ignored)
        ROS_INFO_STREAM("Testing classifier on " << listFilename);
        double resultingAccuracy = testClassifier(listFilename);
        ROS_INFO("Testing complete, average accuracy is %.2f%%!", 100.0 * resultingAccuracy);
    }
    else
    {
        ROS_WARN("_list_file argument was not specified; not testing on any point clouds. This file should contain "
                 "a cloud (PCD) filename + label separated by space or tabulator per line; the first two lines are ignored.");
    }
        
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
