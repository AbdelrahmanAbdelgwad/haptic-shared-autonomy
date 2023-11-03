//Written by Ahmet Burkay KIRNIK
//TR_CapaFenLisesi
//Measure Angle with a MPU-6050(GY-521)

#include<Wire.h>

#include <ros.h>
#include <std_msgs/String.h>
#include <std_msgs/Int16.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int16.h>

ros::NodeHandle nh;
std_msgs::Float32 difference;
std_msgs::Int16 x_imu;

float agent_sig; float sig_diff;

const int MPU_addr=0x68;
int16_t AcX,AcY,AcZ,Tmp,GyX,GyY,GyZ;

int minVal=265;
int maxVal=402;

double x;
double y;
double z;
 
void messageCb( const std_msgs::Float32& msg){
  
  agent_sig = msg.data;
  sig_diff = agent_sig - x;
}

ros::Subscriber<std_msgs::Float32> agent("/agent", &messageCb);
//ros::Publisher diff ("/diff",&difference);
ros::Publisher angle("/rotation",&x_imu);


void setup(){
  Wire.begin();
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);
  Serial.begin(9600);
  
   nh.initNode();
  nh.subscribe(agent);
//  nh.advertise(diff);
    nh.advertise(angle);
}
void loop(){
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_addr,14,true);
  AcX=Wire.read()<<8|Wire.read();
  AcY=Wire.read()<<8|Wire.read();
  AcZ=Wire.read()<<8|Wire.read();
    int xAng = map(AcX,minVal,maxVal,-90,90);
    int yAng = map(AcY,minVal,maxVal,-90,90);
    int zAng = map(AcZ,minVal,maxVal,-90,90);

       x= RAD_TO_DEG * (atan2(-yAng, -zAng)+PI);
       x = map(x,0,360,-180,180);
       y= RAD_TO_DEG * (atan2(-xAng, -zAng)+PI);
       z= RAD_TO_DEG * (atan2(-yAng, -xAng)+PI);

      if (x > 0)
      {
        x = x -180;
        }
        else
        {
          x = x + 180;
        }
        x=constrain(x,-90,90);
//x = (x + 35) * (1 + 1) / (35 + 35) -1 ;
  x_imu.data = round(-x);
//  diff.publish(&difference);
  angle.publish(&x_imu); 
  nh.spinOnce();


     
}
