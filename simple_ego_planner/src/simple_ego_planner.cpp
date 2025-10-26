#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <Eigen/Eigen>
#include <iostream>
#include <vector>
#include <cmath>

class SimpleEgoPlanner
{
public:
    SimpleEgoPlanner(ros::NodeHandle& nh) : nh_(nh)
    {
        // 初始化参数
        initParameters();
        
        // 初始化发布者和订阅者
        initPubSub();
        
        // 初始化状态
        current_state_ = IDLE;
        has_odom_ = false;
        has_target_ = false;
        
        // 创建定时器
        plan_timer_ = nh_.createTimer(ros::Duration(0.05), &SimpleEgoPlanner::planTimerCallback, this);
        
        ROS_INFO("[SimpleEgoPlanner] Initialized successfully!");
    }

private:
    enum PlannerState
    {
        IDLE,           // 空闲状态
        PLANNING,       // 规划中
        EXECUTING       // 执行轨迹
    };

    struct TrajectoryPoint
    {
        Eigen::Vector3d position;
        Eigen::Vector3d velocity;
        Eigen::Vector3d acceleration;
        double yaw;
        double time;
    };

    // ROS相关
    ros::NodeHandle nh_;
    ros::Subscriber odom_sub_;
    ros::Subscriber target_sub_;
    ros::Publisher pos_cmd_pub_;
    ros::Publisher traj_vis_pub_;
    ros::Publisher goal_vis_pub_;
    ros::Timer plan_timer_;

    // 状态变量
    PlannerState current_state_;
    bool has_odom_;
    bool has_target_;
    
    // 当前状态
    Eigen::Vector3d current_pos_;
    Eigen::Vector3d current_vel_;
    Eigen::Vector3d target_pos_;
    
    // 轨迹
    std::vector<TrajectoryPoint> trajectory_;
    int current_traj_index_;
    ros::Time traj_start_time_;
    ros::Time last_goal_time_;
    
    // 参数
    double max_vel_;           // 最大速度
    double max_acc_;           // 最大加速度
    double goal_tolerance_;    // 目标容忍度
    double dt_;               // 时间步长
    double publish_rate_;     // 发布频率
    
    void initParameters()
    {
        nh_.param("max_vel", max_vel_, 1.5);
        nh_.param("max_acc", max_acc_, 2.0);
        nh_.param("goal_tolerance", goal_tolerance_, 0.25);
        nh_.param("dt", dt_, 0.1);
        nh_.param("publish_rate", publish_rate_, 20.0);
        
        ROS_INFO("[SimpleEgoPlanner] Parameters loaded:");
        ROS_INFO("  max_vel: %.2f m/s", max_vel_);
        ROS_INFO("  max_acc: %.2f m/s^2", max_acc_);
        ROS_INFO("  goal_tolerance: %.2f m", goal_tolerance_);
    }
    
    void initPubSub()
    {
        // 订阅者 - 使用与PX4CtrlFSM相同的话题
        odom_sub_ = nh_.subscribe("/mavros/local_position/pose", 1, &SimpleEgoPlanner::odomCallback, this);
        target_sub_ = nh_.subscribe("/move_base_simple/goal", 1, &SimpleEgoPlanner::targetCallback, this);
        
        // 发布者 - 发布到PX4CtrlFSM期望的话题
        pos_cmd_pub_ = nh_.advertise<quadrotor_msgs::PositionCommand>("/planning/pos_cmd", 10);
        traj_vis_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/simple_planner/trajectory", 10);
        goal_vis_pub_ = nh_.advertise<visualization_msgs::Marker>("/simple_planner/goal", 10);
    }
    
    void odomCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        current_pos_ << msg->pose.position.x, 
                        msg->pose.position.y, 
                        msg->pose.position.z;
        
        // 简单的数值微分估计速度
        static Eigen::Vector3d last_pos = current_pos_;
        static ros::Time last_time = ros::Time::now();
        
        ros::Time current_time = ros::Time::now();
        double dt = (current_time - last_time).toSec();
        
        if (dt > 0.001 && has_odom_) {
            current_vel_ = (current_pos_ - last_pos) / dt;
        }
        
        last_pos = current_pos_;
        last_time = current_time;
        has_odom_ = true;
    }
    
    void targetCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        target_pos_ << msg->pose.position.x,
                       msg->pose.position.y,
                       msg->pose.position.z;
        
        has_target_ = true;
        current_state_ = PLANNING;
        last_goal_time_ = ros::Time::now();
        
        ROS_INFO("[SimpleEgoPlanner] New target received: (%.2f, %.2f, %.2f)", 
                 target_pos_.x(), target_pos_.y(), target_pos_.z());
        
        // 可视化目标点
        publishGoalVisualization();
    }
    
    void planTimerCallback(const ros::TimerEvent& /*event*/)
    {
        if (!has_odom_ || !has_target_)
            return;
            
        switch (current_state_)
        {
            case IDLE:
                break;
                
            case PLANNING:
                if (planTrajectory())
                {
                    current_state_ = EXECUTING;
                    current_traj_index_ = 0;
                    traj_start_time_ = ros::Time::now();
                    ROS_INFO("[SimpleEgoPlanner] Trajectory planned, %lu points", trajectory_.size());
                }
                else
                {
                    ROS_WARN("[SimpleEgoPlanner] Trajectory planning failed");
                    current_state_ = IDLE;
                }
                break;
                
            case EXECUTING:
                if (isGoalReached())
                {
                    current_state_ = IDLE;
                    has_target_ = false;
                    ROS_INFO("[SimpleEgoPlanner] Goal reached!");
                    // 发布停止命令
                    publishStopCommand();
                }
                else
                {
                    executeTrajectory();
                }
                break;
        }
    }
    
    bool planTrajectory()
    {
        trajectory_.clear();
        
        // 检查是否已经到达目标
        double distance = (target_pos_ - current_pos_).norm();
        if (distance < goal_tolerance_)
        {
            return false;
        }
        
        // 简化的轨迹规划：直线轨迹 + 梯形速度曲线
        Eigen::Vector3d direction = (target_pos_ - current_pos_).normalized();
        
        // 计算运动时间
        double total_time = calculateOptimalTime(distance);
        
        // 生成轨迹点
        int num_points = static_cast<int>(total_time / dt_) + 1;
        trajectory_.reserve(num_points);
        
        for (int i = 0; i < num_points; ++i)
        {
            double t = i * dt_;
            if (t > total_time) t = total_time;
            
            TrajectoryPoint point;
            
            // 使用梯形速度曲线计算位置、速度、加速度
            calculateTrajectoryPoint(t, total_time, distance, direction, point);
            
            trajectory_.push_back(point);
        }
        
        // 添加最终点确保到达目标
        if (!trajectory_.empty()) {
            TrajectoryPoint final_point;
            final_point.position = target_pos_;
            final_point.velocity = Eigen::Vector3d::Zero();
            final_point.acceleration = Eigen::Vector3d::Zero();
            final_point.yaw = 0.0;
            final_point.time = total_time;
            trajectory_.push_back(final_point);
        }
        
        // 可视化轨迹
        publishTrajectoryVisualization();
        
        return !trajectory_.empty();
    }
    
    double calculateOptimalTime(double distance)
    {
        // 梯形速度曲线的时间计算
        double acc_time = max_vel_ / max_acc_;
        double acc_dist = 0.5 * max_acc_ * acc_time * acc_time;
        
        if (2 * acc_dist >= distance)
        {
            // 三角形速度曲线
            return 2.0 * sqrt(distance / max_acc_);
        }
        else
        {
            // 梯形速度曲线
            double const_vel_dist = distance - 2 * acc_dist;
            return 2 * acc_time + const_vel_dist / max_vel_;
        }
    }
    
    void calculateTrajectoryPoint(double t, double total_time, double total_distance, 
                                  const Eigen::Vector3d& direction, TrajectoryPoint& point)
    {
        double acc_time = max_vel_ / max_acc_;
        double acc_dist = 0.5 * max_acc_ * acc_time * acc_time;
        
        double s, v, a;  // 位置、速度、加速度标量值
        
        if (2 * acc_dist >= total_distance)
        {
            // 三角形速度曲线
            double peak_time = total_time / 2.0;
            if (t <= peak_time)
            {
                s = 0.5 * max_acc_ * t * t;
                v = max_acc_ * t;
                a = max_acc_;
            }
            else
            {
                double dt_from_peak = t - peak_time;
                double peak_vel = max_acc_ * peak_time;
                s = acc_dist + peak_vel * dt_from_peak - 0.5 * max_acc_ * dt_from_peak * dt_from_peak;
                v = peak_vel - max_acc_ * dt_from_peak;
                a = -max_acc_;
            }
        }
        else
        {
            // 梯形速度曲线
            if (t <= acc_time)
            {
                s = 0.5 * max_acc_ * t * t;
                v = max_acc_ * t;
                a = max_acc_;
            }
            else if (t <= total_time - acc_time)
            {
                s = acc_dist + max_vel_ * (t - acc_time);
                v = max_vel_;
                a = 0.0;
            }
            else
            {
                double dt_from_decel = t - (total_time - acc_time);
                s = total_distance - acc_dist + max_vel_ * dt_from_decel - 0.5 * max_acc_ * dt_from_decel * dt_from_decel;
                v = max_vel_ - max_acc_ * dt_from_decel;
                a = -max_acc_;
            }
        }
        
        // 转换为3D向量
        point.position = current_pos_ + direction * s;
        point.velocity = direction * std::max(0.0, v);
        point.acceleration = direction * a;
        point.yaw = 0.0;  // 简化版不考虑偏航
        point.time = t;
    }
    
    void executeTrajectory()
    {
        if (trajectory_.empty()) return;
        
        double current_time = (ros::Time::now() - traj_start_time_).toSec();
        
        // 找到当前应该执行的轨迹点
        while (current_traj_index_ < trajectory_.size() - 1 && 
               trajectory_[current_traj_index_ + 1].time <= current_time)
        {
            current_traj_index_++;
        }
        
        if (current_traj_index_ >= trajectory_.size() - 1)
        {
            current_traj_index_ = trajectory_.size() - 1;
        }
        
        // 发布当前轨迹点
        const TrajectoryPoint& point = trajectory_[current_traj_index_];
        publishPositionCommand(point);
    }
    
    void publishPositionCommand(const TrajectoryPoint& point)
    {
        quadrotor_msgs::PositionCommand cmd;
        
        cmd.header.stamp = ros::Time::now();
        cmd.header.frame_id = "map";
        
        // 位置
        cmd.position.x = point.position.x();
        cmd.position.y = point.position.y();
        cmd.position.z = point.position.z();
        
        // 速度
        cmd.velocity.x = point.velocity.x();
        cmd.velocity.y = point.velocity.y();
        cmd.velocity.z = point.velocity.z();
        
        // 加速度
        cmd.acceleration.x = point.acceleration.x();
        cmd.acceleration.y = point.acceleration.y();
        cmd.acceleration.z = point.acceleration.z();
        
        // 偏航角
        cmd.yaw = point.yaw;
        cmd.yaw_dot = 0.0;
        
        pos_cmd_pub_.publish(cmd);
    }
    
    void publishStopCommand()
    {
        quadrotor_msgs::PositionCommand cmd;
        
        cmd.header.stamp = ros::Time::now();
        cmd.header.frame_id = "map";
        
        // 当前位置，零速度和加速度
        cmd.position.x = current_pos_.x();
        cmd.position.y = current_pos_.y();
        cmd.position.z = current_pos_.z();
        
        cmd.velocity.x = 0.0;
        cmd.velocity.y = 0.0;
        cmd.velocity.z = 0.0;
        
        cmd.acceleration.x = 0.0;
        cmd.acceleration.y = 0.0;
        cmd.acceleration.z = 0.0;
        
        cmd.yaw = 0.0;
        cmd.yaw_dot = 0.0;
        
        pos_cmd_pub_.publish(cmd);
    }
    
    bool isGoalReached()
    {
        return (current_pos_ - target_pos_).norm() < goal_tolerance_;
    }
    
    void publishTrajectoryVisualization()
    {
        visualization_msgs::MarkerArray marker_array;
        
        // 轨迹线
        visualization_msgs::Marker line_marker;
        line_marker.header.frame_id = "map";
        line_marker.header.stamp = ros::Time::now();
        line_marker.ns = "trajectory";
        line_marker.id = 0;
        line_marker.type = visualization_msgs::Marker::LINE_STRIP;
        line_marker.action = visualization_msgs::Marker::ADD;
        line_marker.scale.x = 0.05;
        line_marker.color.r = 0.0;
        line_marker.color.g = 1.0;
        line_marker.color.b = 0.0;
        line_marker.color.a = 0.8;
        
        for (const auto& point : trajectory_)
        {
            geometry_msgs::Point p;
            p.x = point.position.x();
            p.y = point.position.y();
            p.z = point.position.z();
            line_marker.points.push_back(p);
        }
        
        marker_array.markers.push_back(line_marker);
        traj_vis_pub_.publish(marker_array);
    }
    
    void publishGoalVisualization()
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = ros::Time::now();
        marker.ns = "goal";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
        
        marker.pose.position.x = target_pos_.x();
        marker.pose.position.y = target_pos_.y();
        marker.pose.position.z = target_pos_.z();
        marker.pose.orientation.w = 1.0;
        
        marker.scale.x = 0.5;
        marker.scale.y = 0.5;
        marker.scale.z = 0.5;
        
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 0.8;
        
        goal_vis_pub_.publish(marker);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "simple_ego_planner");
    ros::NodeHandle nh("~");
    
    SimpleEgoPlanner planner(nh);
    
    ROS_INFO("[SimpleEgoPlanner] Node started. Use 2D Nav Goal in RViz to set target.");
    ROS_INFO("[SimpleEgoPlanner] Publishing to /planning/pos_cmd for PX4CtrlFSM");
    
    ros::spin();
    
    return 0;
}
