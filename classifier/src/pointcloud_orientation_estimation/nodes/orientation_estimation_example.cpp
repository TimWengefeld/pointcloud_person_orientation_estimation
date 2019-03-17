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
#include <geometry_msgs/PoseStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

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
#include "../classifier.h"
#include "../features.h"

using namespace std;

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <chrono>
#define foreach BOOST_FOREACH

#include <signal.h>
#include <ros/xmlrpc_manager.h>


// Global variables
ros::Publisher g_pointCloudPublisher;
ros::Publisher g_posePublisher;
std::vector<boost::shared_ptr<TopDownClassifier> > g_multiclass_classifier;

// Signal-safe flag for whether shutdown is requested
sig_atomic_t volatile g_request_shutdown = 0;
// Replacement SIGINT handler
void mySigIntHandler(int sig)
{
  g_request_shutdown = 1;
}

struct CloudInfo {
    pcl::PointXYZ pose, velocity;
    float thetaDeg; // person orientation. 180° = looking INTO camrea
    float sensorDistance;
    float phi; // angle between optical axis of camera and person
    size_t numPoints;
};

bool loadCloudInfo(std::string poseFilename, CloudInfo& cloudInfo)
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

    poseFile >> fieldName >> cloudInfo.pose.x;            if(poseFile.fail()) { poseFile.clear(); poseFile >> NaNString; }  ROS_ASSERT(fieldName == "position_x");
    poseFile >> fieldName >> cloudInfo.pose.y;            if(poseFile.fail()) { poseFile.clear(); poseFile >> NaNString; }  ROS_ASSERT(fieldName == "position_y");
    poseFile >> fieldName >> cloudInfo.velocity.x;        if(poseFile.fail()) { poseFile.clear(); poseFile >> NaNString; }  ROS_ASSERT(fieldName == "velocity_x");
    poseFile >> fieldName >> cloudInfo.velocity.y;        if(poseFile.fail()) { poseFile.clear(); poseFile >> NaNString; }  ROS_ASSERT(fieldName == "velocity_y");
    poseFile >> fieldName >> cloudInfo.thetaDeg;          if(poseFile.fail()) { poseFile.clear(); poseFile >> NaNString; }  ROS_ASSERT(fieldName == "theta_deg");
    poseFile >> fieldName >> cloudInfo.sensorDistance;    if(poseFile.fail()) { poseFile.clear(); poseFile >> NaNString; }  ROS_ASSERT(fieldName == "sensor_distance");

    cloudInfo.phi = atan2(cloudInfo.pose.y, cloudInfo.pose.x) * 180.0 / M_PI;
    return true;
}

// Replacement "shutdown" XMLRPC callback
void shutdownCallback(XmlRpc::XmlRpcValue& params, XmlRpc::XmlRpcValue& result)
{
  int num_params = 0;
  if (params.getType() == XmlRpc::XmlRpcValue::TypeArray)
    num_params = params.size();
  if (num_params > 1)
  {
    std::string reason = params[1];
    ROS_WARN("Shutdown request received. Reason: [%s]", reason.c_str());
    g_request_shutdown = 1; // Set flag
  }

  result = ros::xmlrpc::responseInt(1, "", 0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
string estimateOrientations(const std::string& cloudFilename)
{  

    // Load point cloud from disk
    PointCloud::Ptr personCloud(new PointCloud);
    ROS_DEBUG_STREAM("Loading " << cloudFilename << "...");

    if(pcl::io::loadPCDFile<PointType>(cloudFilename, *personCloud) == -1)
    {
        ROS_ERROR("Couldn't read file %s\n", cloudFilename.c_str());
        exit(-1);
    }
	personCloud->header.frame_id = "extracted_cloud_frame";
	personCloud->header.stamp = ros::Time::now().toNSec() / 1000;
	g_pointCloudPublisher.publish(personCloud);
	ros::WallRate rate(25);

	double maxPrediction = -std::numeric_limits<double>::infinity();
	string maxLabel;

    std::chrono::duration<double, milli> duration(0);
    std::chrono::duration<double, milli> durationFeatures(0);
    std::chrono::duration<double, milli> durationClassify(0);

    for(int i = 0; i < g_multiclass_classifier.size();i++)
	{
		cv::Mat featureVector, missingDataMask;
		auto startFeatures = std::chrono::high_resolution_clock::now();
		g_multiclass_classifier[i]->calculateFeatures(*personCloud,featureVector, missingDataMask);
		auto endFeatures = std::chrono::high_resolution_clock::now();
		auto startClassify = std::chrono::high_resolution_clock::now();
		double sumOfVotes;
		g_multiclass_classifier[i]->classifyFeatureVector(featureVector,missingDataMask,&sumOfVotes); // invoke classifier
		auto endClassify = std::chrono::high_resolution_clock::now();
		durationFeatures += endFeatures-startFeatures;
		durationClassify += endClassify-startClassify;

		if(sumOfVotes > maxPrediction)
		{
				maxPrediction = sumOfVotes;
				maxLabel = g_multiclass_classifier[i]->getCategory();
		}
	}

    std::cout << "Estimation process without loading from disk took " << endl;
    std::cout << durationFeatures.count() << " ms for feature extraction" << endl;
    std::cout << durationClassify.count() << " ms for classification" << endl;

    tf2::Quaternion myQuaternion;
    myQuaternion.setRPY( 0, 0, (std::stof(maxLabel)+180) / 180.0 * M_PI );
    geometry_msgs::PoseStamped poseStamped;
    poseStamped.header.frame_id="extracted_cloud_frame";
    poseStamped.header.stamp = ros::Time::now();
    poseStamped.pose.position.x = 0;
    poseStamped.pose.position.y = 0;
    poseStamped.pose.position.z = 1.0;
    poseStamped.pose.orientation.x = myQuaternion.getX();
    poseStamped.pose.orientation.y = myQuaternion.getY();
    poseStamped.pose.orientation.z = myQuaternion.getZ();
    poseStamped.pose.orientation.w = myQuaternion.getW();
    g_posePublisher.publish(poseStamped);

    return maxLabel;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    ros::init(argc, argv, "classify_single_cloud", ros::init_options::NoSigintHandler);
    signal(SIGINT, mySigIntHandler);

    // Override XMLRPC shutdown
    ros::XMLRPCManager::instance()->unbind("shutdown");
    ros::XMLRPCManager::instance()->bind("shutdown", shutdownCallback);

    ros::NodeHandle nodeHandle("");
    ros::NodeHandle privateHandle("~");

    //
    // Parse arguments
    //

    bool showBestTessellation; std::string exampleFilesPath, modelFilename; int numThreads;
    privateHandle.param<std::string>("exampleFilesPath", exampleFilesPath, "");
    privateHandle.param<std::string>("model", modelFilename, "");
    privateHandle.param<int>("num_threads", numThreads, 5);

    omp_set_num_threads(numThreads);
    ROS_INFO_STREAM("Using " << numThreads << " parallel threads for feature computations!");
    
    // Create point cloud publisher
    g_pointCloudPublisher = nodeHandle.advertise<sensor_msgs::PointCloud2>("cloud", 1, true);
    g_posePublisher = nodeHandle.advertise<geometry_msgs::PoseStamped>("pose", 1, true);
    
    
    //
    // Load classifier
    //

    if(modelFilename.empty()) {
        ROS_ERROR_STREAM("The _model argument was not specified; this is required, and must point to a YAML file containing the learned classifier.");
        return -1;
    }

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
    for(int i = 0; i < mClassifierPaths.size(); i++)
	{
		g_multiclass_classifier.push_back(boost::shared_ptr<TopDownClassifier>(new TopDownClassifier));
		std::cout << "load model from : " << mClassifierPaths[i] << std::endl;
		g_multiclass_classifier.back()->init(mClassifierPaths[i]);
	}

    //
    // Classify cloud
    //
    if(boost::filesystem::is_directory(exampleFilesPath))
    {
    	std::map<string,float> pathStrings;
    	for(auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(exampleFilesPath), {}))
		{
			if(entry.path().extension() == ".txt")
			{
				CloudInfo cloudInfo;
				loadCloudInfo(entry.path().string(), cloudInfo);
				std::string pcdPath = entry.path().string();
				boost::replace_all(pcdPath,"_pose.txt","_cloud.pcd");
				pathStrings[pcdPath] = cloudInfo.thetaDeg;
			}
		}
    	while(!g_request_shutdown)
    	{
    		for(auto const& path : pathStrings)
    		{
				if(!g_request_shutdown)
				{
					string label = estimateOrientations(path.first);
					std::cout << "estimated orientation : " << label << std::endl;
					std::cout << "ground truth label was : " << path.second << std::endl << std::endl;
				}
    		}
    	}
    }
    else
    {
        ROS_ERROR("_exampleFilesPath argument has not been specified!");
        return -1;
    }
        
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
