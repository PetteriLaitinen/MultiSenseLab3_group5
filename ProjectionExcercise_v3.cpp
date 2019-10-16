#include "opencv2/opencv.hpp"
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <experimental/filesystem>
namespace fs =  std::experimental::filesystem::v1;

using namespace cv;

// Read next video timestamp to make sure the video is synchronized
double read_next_video_timestamp(std::ifstream &file){
  std::string str;
  std::getline(file, str);

  std::string data = str.substr(str.find(':')+2, str.find(',')-(str.find(':')+2));
  double hr = std::stof(data.substr(0, data.find(':')));
  data.erase(0, data.find(':')+1);
  double min = std::stof(data.substr(0, data.find(':')));
  data.erase(0, data.find(':')+1);
  double sec = std::stof(data);

  sec = (hr*60+min)*60+sec;

  return sec;
};

//read points into point_array until the given timestamp is passed (minimum 1 line)
int read_until(std::ifstream &file, std::vector<cv::Point3d> &point_array, double timestamp, std::vector<std::string> &lidar_str){
  std::string str;
  double new_timestamp = 0.0;
  double lo = timestamp - new_timestamp;
  int ctr = 0; 
  while (lo > 0.0 )
  {
    std::getline(file, str);
    //const char* str = istr.c_str();
    new_timestamp = std::stod( str.substr( 1, str.find(':') ) );
    
    std::string data = str.substr(str.find(':')+1, str.find(']')-1);
    while(data.length() > 1){

      std::string entry = data.substr(0,data.find(';'));
      
      double d = std::stof(entry.substr(0,entry.find('|')));
      entry.erase(0, entry.find('|')+1);
      double h = std::stof(entry.substr(0,entry.find('|')));
      entry.erase(0, entry.find('|')+1);
      double v = std::stof(entry.substr(0,entry.find('|')));

      data.erase(0, data.find(';')+1);
      
      double diag = d * cos(v);

      point_array.push_back( Point3d( diag * cos(h), diag * sin(h), d * -sin(v) ) );

      lidar_str.push_back(str);
    }
    ctr++;
    lo = timestamp-new_timestamp;   
  }
  return ctr;
};

//read objects into an array until the timestamp is passed (minimum 1 line)
int read_obj_until(std::ifstream &file, std::vector<cv::Vec4d> &object_array, double timestamp, std::vector<cv::Point3d> &extra_params){
  std::string str;
  double new_timestamp = 0.0;
  double lo = timestamp - new_timestamp;
  int ctr = 0; 
  while (lo > 0.0 )
  {
    if(file.eof()){ return -1; }
    std::getline(file, str);
    new_timestamp = std::stod( str.substr( 1, str.find(':') ) );
    
    std::string data = str.substr(str.find(':')+1, str.find(']')-1);
    while(data.length() > 1){

      std::string entry = data.substr(0,data.find(';'));
      
      double center_x = std::stof(entry.substr(1,data.find(',')));
      entry.erase(0, entry.find(',')+2);
      double center_y = std::stof(entry.substr(0,entry.find(')')));
      entry.erase(0, entry.find(')')+2);

      //first 2 extra parameters: velocity x and velocity y      
      double vel_x = std::stof(entry.substr(1,data.find(',')));
      entry.erase(0, entry.find(',')+2);
      double vel_y = std::stof(entry.substr(0,entry.find(')')));
      entry.erase(0, entry.find(')')+2);

      //remove second velocity parameter
      entry.erase(0, entry.find(')')+2);

      //read width and height
      double width = std::stof(entry.substr(1,data.find(',')));
      entry.erase(0, entry.find(',')+2);
      double height = std::stof(entry.substr(0,entry.find(')')));
      entry.erase(0, entry.find(')')+2);     

      int age= std::stoi(entry.substr(0,entry.find('|')));
      
      data.erase(0, data.find(';')+1);

      object_array.push_back( cv::Vec4d( center_x - width/2.0,center_y-height/2.0, center_x + width/2.0,center_y + height/2.0) );
      extra_params.push_back( Point3d( vel_x, vel_y, age ) );
    }
    ctr++;
    lo = timestamp-new_timestamp;   
  }
  return ctr;
};


// Calculates rotation matrix given euler angles.
cv::Mat eulerAnglesToRotationMatrix(double* theta)
{
    // Calculate rotation about x axis
    cv::Mat R_x = (cv::Mat_<double>(3,3) <<
               1,       0,              0,
               0,       cos(theta[0]),   -sin(theta[0]),
               0,       sin(theta[0]),   cos(theta[0])
               );
     
    // Calculate rotation about y axis
    cv::Mat R_y = (Mat_<double>(3,3) <<
               cos(theta[1]),    0,      sin(theta[1]),
               0,               1,      0,
               -sin(theta[1]),   0,      cos(theta[1])
               );
     
    // Calculate rotation about z axis
    cv::Mat R_z = (Mat_<double>(3,3) <<
               cos(theta[2]),    -sin(theta[2]),      0,
               sin(theta[2]),    cos(theta[2]),       0,
               0,               0,                  1);
     
     
    // Combined rotation matrix
    cv::Mat R = R_z * R_y * R_x;
     
    return R;
 
}


// Return true if 2 given boxes overlap anywhere
bool overlap(cv::Vec4d obj1, cv::Vec4d obj2){
    return !(obj1[0] > obj2[2] || obj2[0] > obj1[2] || obj1[1] < obj2[3] || obj2[1] < obj1[3]);
}

//Return the distance between point and a given rectangle
double distance(cv::Vec4d rect, cv::Point3d p) {
  double dx = std::max(rect[0] - p.x, p.x - rect[2]);
  dx = std::max(dx, 0.0);
  double dy = std::max(rect[1] - p.y, p.y - rect[3]);
  dy = std::max(dy, 0.0);
  return std::sqrt(dx*dx + dy*dy);
}

//Fuses all boxes in the given array that overlap
void fuse_similar_objects(std::vector<cv::Vec4d> objects){
  for(int i = 0; i < objects.size(); i++){
    for(int j = i+1; j < objects.size(); j++){
      if(overlap(objects[i],objects[j])){
        objects[i][0] = std::min(objects[i][0],objects[j][0]);
	objects[i][1] = std::min(objects[i][1],objects[j][1]);
	objects[i][2] = std::min(objects[i][0],objects[j][0]);
	objects[i][3] = std::min(objects[i][1],objects[j][1]);
	objects.erase(objects.begin()+j);
	i = 0;
	j = 0;
      }
    }
  }
}

int main(int, char**)
{
  std::cout << "let's go!\n";
  //Lidar rotation relative to camera
  double rv [3];
  rv[0] = 0;//-0.012;
  rv[1] = -0.018;//-0.062;
  rv[2] = 0.002;//0.116;

  //Lidar translation relative to camera
  cv::Mat tvec(3,1,cv::DataType<double>::type);
  tvec.at<double>(0) = -0.1;//0.55;
  tvec.at<double>(1) = -0.25;//-0.05;
  tvec.at<double>(2) = 0;//-0.05;

  //Distortion matrix for the camera
  cv::Mat K (3,3,cv::DataType<double>::type);
  K.at<double>(0,0) = 674*2; 
  K.at<double>(0,1) = 0.;
  K.at<double>(0,2) = 320*2;//960.;

  K.at<double>(1,0) = 0.;
  K.at<double>(1,1) = 674*1.5;
  K.at<double>(1,2) = 240*1.5;//540;

  K.at<double>(2,0) = 0.;
  K.at<double>(2,1) = 0.;
  K.at<double>(2,2) = 1.;

  cv::Mat distCoeffs(5,1,cv::DataType<double>::type);
  distCoeffs.at<double>(0) = -0.4662177;
  distCoeffs.at<double>(1) = 0.1638718;
  distCoeffs.at<double>(2) = 0.;
  distCoeffs.at<double>(3) = 0.;
  distCoeffs.at<double>(4) = 0.2586795;//-1.3972235806313755e+00;

  //Matrix for relative rotation of lidar and camera
  cv::Mat rvecR(3,3,cv::DataType<double>::type);
  cv::Mat baseR(3,3,cv::DataType<double>::type);
  baseR.at<double>(0,0) = 0.;
  baseR.at<double>(1,0) = 0.;
  baseR.at<double>(2,0) = 1.;
  baseR.at<double>(0,1) = -1.;
  baseR.at<double>(1,1) = 0.;
  baseR.at<double>(2,1) = 0.;
  baseR.at<double>(0,2) = 0.;
  baseR.at<double>(1,2) = -1.;
  baseR.at<double>(2,2) = 0.;
  cv::Mat er = eulerAnglesToRotationMatrix(rv);
  rvecR = baseR*er;
  cv::Mat rvec(3,1,cv::DataType<double>::type);
  cv::Rodrigues(rvecR,rvec);

  //Initialization of point arrays

  //Current frame for point projection
  Mat baseImg;
  //Points from lidar
  std::vector<cv::Point3d> points; 
  //Projected points from lidar
  std::vector<cv::Point2d> projection_points;
  //Distances to points from lidar
  std::vector<double> distances;
  //Objects as detected by lidar
  std::vector<cv::Vec4d> objects;
  //Projected objects in opencv format
  std::vector<cv::Rect> projection_objects;

  //Excercise - points closest to rectangles detected by lidar
  std::vector<int> *closestPoints;
  
  //Some extra paremeters
  std::vector<cv::Point3d> extra_pars;
  std::vector<std::string> lpoints_str;

  //current timestamp
  double timestamp = 0.0;
  double prev_stamp = -100.0;
  //whether to play the video
  int play = 1;
  //in paused mode tells to read the next frame
  int nf = 0;
  
  //Files to use for point projection
  std::ifstream file("./lidarPoints.txt");
  std::ifstream ofile("./lidarObjects.txt");
  std::ifstream tfile("./lidarTimestamps.txt");
  VideoCapture cap("./lidarVideo.mp4");

  if(!cap.isOpened()){ 
    return -1; // check if we succeeded 
  }

  //open lidar logs and read 2 lines from them before starting
  std::string str; 
  std::string ostr;
  std::getline(file, str);
  std::getline(ofile, str);
  str = "";
  ostr = "";
  std::getline(ofile, ostr);
  std::getline(file, str);

  //Configure offset, as zero for time for camera and lidar are different
  double orig_start = 0.0; //std::stod(str.substr(1,str.find(":")));
  double offset = -7.4;
  double next_timestamp = orig_start;      

  //Initialize window for showing data
  namedWindow("fusion",1);

  
  for(;;)
  {
    Mat frame;
    //If the video is playing, read the next frame
    if(play || nf){
      nf = 0;
      cap >> frame;
      frame.copyTo(baseImg);
      timestamp = read_next_video_timestamp(tfile)+orig_start;
    } else {
      baseImg.copyTo(frame);
    } 
 
    //Clear the arrays and read new data into them when the timestamps are passed
    if (timestamp >= next_timestamp){  
      if(timestamp>(orig_start-offset) && prev_stamp < (timestamp+offset)){
        points.clear();
	objects.clear();
	extra_pars.clear();
	lpoints_str.clear();
 	int sz = read_until(file, points, timestamp+offset, lpoints_str);
	if(sz < 0){ return 0; }
	read_obj_until(ofile, objects, timestamp+offset, extra_pars);
        prev_stamp = timestamp+offset;
      }
      next_timestamp = timestamp + 0.2;        
    }
    if (points.size() > 0 & objects.size() > 0) {
      //std::cout << "p & o: " << points.size() << " " << objects.size() <<"\n";
    }
    //If there are Lidar points to parse
    if(points.size() > 0){
      //Clear the arrays
      projection_points.clear();
      projection_objects.clear();
      distances.clear();
      //Project the 3D Lidar points onto a 2D pane, that can be shown on a picture 
      projectPoints( points, rvec, tvec, K, distCoeffs, projection_points );
      //For each point add it's distance to a separate array
      for(int i =0; i < points.size();i++){
        distances.push_back(points[i].x);
      }

      //if lidar has detected objects
      if(objects.size()> 0){
        //fuse similar objects together
        fuse_similar_objects(objects);
        //for(int i = 0; i < objects.size(); i++){
        //  projection_objects.push_back(cv::Rect(0,0,0,0));
        //}
	//Excercise
	int sz = objects.size();
	//std::cout << sz << "  ";

	closestPoints = new std::vector<int>[sz];
	int smallestDist = 99999999;
	int currDist = 0;
	int closestRect = 0;
        for(int i =0; i < points.size();i++){
          for(int j = 0; j < objects.size(); j++){
	    currDist = distance(objects[j], points[i]);
	    if (currDist < smallestDist) {
	      smallestDist = currDist;
	      closestRect = j;
	    }
	  }
	  closestPoints[closestRect].push_back(i);
	  //std::cout << "Lähin piste lisätty\n";
      	}
      	//std::cout << closestPoints.size();	
        //PROBLEM: Lidar detects objects as 2d objects from top-down view, so it will not tell how high it is. 
	//For visualizing objects you need to determine which point each object belongs to (use distance() function) and draw a box around those points.
	//This box needs to be saved in Rect object in projection_objects list
        //TODO: For every box in the objects list, you should have a box in projection_objects list that is over it.
	for(int i=0; i < sz; i++) {
	  // if there are two or more points we can form a rectangle
	  if(closestPoints[i].size() > 1) {
	    int minX = 999999;
	    int minY = 999999;
	    int maxX = -1;
	    int maxY = -1;
	    bool added = false;
	    for(int j=0; j < closestPoints[i].size(); j++) {
	      //Should values be tested not to be negative or out of drawing window?
	      int px = projection_points[closestPoints[i][j]].x;
	      int py = projection_points[closestPoints[i][j]].y;
	      if (px < 5 | px > 1275) continue;
	      if (py < 5 | py > 715) continue;
	      if (px < minX) minX = px;  
	      if (py < minY) minY = py;
	      if (px > maxX) maxX = px;  
	      if (py > maxY) maxY = py;
	      added = true;
	    }
	    if (added) {
	      //Enlarge the rectangle by decreasing x & y by 5 and increasing width and height by 10
	      projection_objects.push_back(cv::Rect(minX-5,minY-5,maxX-minX+10,maxY-minY+10));
	      std::cout << "Neliö lisätty\n";
	    }
	  }
	}
      }
      //std::cout << projection_objects.size();
      points.clear();
      objects.clear();
    }
    //For each projected point, draw it and the distance on top of it.
    std::vector<cv::Rect> textBoxes;
    for(int i=0; i < projection_points.size(); i++){
      if(projection_points[i].x < 0 || projection_points[i].x > 1280 || projection_points[i].y < 0 || projection_points[i].y > 720){ continue; }
      int color = distances[i]*255/150;
      if(color > 255){ color = 255;}
      //Do not draw points that are closer than 3 meters, since those are likely false positives
      if(color > 3){
        circle(frame, projection_points[i], 2, Scalar( color,0,255-color ),2);

	std::string dist = std::to_string(int(distances[i]));
	Point p = Point(projection_points[i].x+3, projection_points[i].y-5);
        /*

        //Check whether the point is too close to another point with drawn distance. This way we don't draw distances on top of each other
	bool printDist = true;
	for(int j = 0; j < textBoxes.size(); j++){
	  if(textBoxes[j].contains(p)){
	    printDist = false;
	    break;
	  }
	}
        //If there is room, draw distance on top of the point
	if(printDist){
	  putText(frame, dist, p, fontFace, 1.3, Scalar( 0,255,0 ), 2, 8);
	  textBoxes.push_back(cv::Rect(p.x-35, p.y-17, 70,35));
	}

        */
      }
    }
    //For each projected object, draw a rectangle over it.
    for(int i=0; i < projection_objects.size(); i++){
      //std::cout << "Neliöitä tarkastelemassa\n";
      //TODO: Implement this function. Take example from how points are drawn and use rectangle() function to draw objects.
      //Look above how distance to points is calculated and display the distance to the object in question (average distance to all points of that object)
      //Boxes should be slightly bigger than the actual box containing points (approximately +10 pixels) as that is better for viewing (looks cleaner)
      //Exercise
      //if(projection_objects[i].x < 5 || projection_objects[i].x + projection_objects[i].width > 1275 || projection_objects[i].y < 5 || projection_objects[i].y + projection_objects[i].height > 715){
        //std::cout << "Ulkona ruudusta\n";
        //continue;
      //}
      int color = distances[i]*255/150;
      if(color > 255){ color = 255;}
      //Do not draw points that are closer than 3 meters, since those are likely false positives
      if(color > 3){
        std::cout << "Neliöitä piirtämässä: " << projection_objects[i].x << " " << projection_objects[i].y;
        std::cout << " "  << projection_objects[i].width << " " << projection_objects[i].height << "\n";
        rectangle(frame, projection_objects[i], Scalar( color,0,255-color ),2);

	std::string dist = std::to_string(int(distances[i]));
	//Point p = Point(projection_points[i].x+3, projection_points[i].y-5);
      }
    }
   
    imshow("fusion", frame);
    
    if(play){
      int key = waitKey(30);
      switch(key){
        case 'v':
	  offset -= 0.2;
	  std::cout<<"new offset: "<<(offset)<<std::endl;  
          break;
        case 'b':
	  offset += 0.2;
	  std::cout<<"new offset: "<<(offset)<<std::endl;  
          break;
        case 32:
	  play = 0;
          break;
        case 27:
          return 0;
      }
    } else{
      int waitingForKey = 1;
      while(waitingForKey){
  	    int key = waitKey(30);
	    switch(key){
	      case 32:
		waitingForKey = 0;
	  	play = 1;
          	break;
	      case 'n':
	      	waitingForKey = 0;
		nf = 1;
	        break;
	      case 27:
	        return 0;
	    }
	}
      }    
  }
  return 0;
}



