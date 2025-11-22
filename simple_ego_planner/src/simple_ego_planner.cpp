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
    double last_yaw_ = 0.0;  // 记录最近一次发布的偏航角
    
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
        nh_.param("goal_tolerance", goal_tolerance_, 0.5);
        nh_.param("dt", dt_, 0.1);
        nh_.param("publish_rate", publish_rate_, 20.0);
        
        ROS_INFO("[SimpleEgoPlanner] Parameters loaded:");
        ROS_INFO("  max_vel: %.2f m/s", max_vel_);
        ROS_INFO("  max_acc: %.2f m/s^2", max_acc_);
        ROS_INFO("  goal_tolerance: %.2f m", goal_tolerance_);
    }

    // 将角度归一化到 [-pi, pi]
    static double normalizeAngle(double ang)
    {
        while (ang > M_PI) ang -= 2.0 * M_PI;
        while (ang < -M_PI) ang += 2.0 * M_PI;
        return ang;
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
        
        // 两段轨迹规划：
        // 第一段：保持当前高度，在xy平面上移动到目标点正上方
        // 第二段：垂直下降到目标点
        
        // 第一段：水平移动
        Eigen::Vector3d horizontal_start = current_pos_;
        Eigen::Vector3d horizontal_target(target_pos_.x(), target_pos_.y(), current_pos_.z());
        double horizontal_distance = (horizontal_target - horizontal_start).norm();
        
        // 第二段：垂直移动
        Eigen::Vector3d vertical_start = horizontal_target;
        Eigen::Vector3d vertical_target = target_pos_;
        double vertical_distance = std::abs(vertical_target.z() - vertical_start.z());
        
        // 生成第一段轨迹（水平移动）
        if (horizontal_distance > goal_tolerance_)
        {
            generateHorizontalTrajectory(horizontal_start, horizontal_target, horizontal_distance);
        }
        
        // 生成第二段轨迹（垂直移动）
        if (vertical_distance > goal_tolerance_)
        {
            generateVerticalTrajectory(vertical_start, vertical_target, vertical_distance);
        }
        
        // 如果没有生成轨迹点，直接到目标
        if (trajectory_.empty())
        {
            TrajectoryPoint final_point;
            final_point.position = target_pos_;
            final_point.velocity = Eigen::Vector3d::Zero();
            final_point.acceleration = Eigen::Vector3d::Zero();
            final_point.yaw = last_yaw_;
            final_point.time = 0.0;
            trajectory_.push_back(final_point);
        }
        else
        {
            // 确保最后一个点到达目标位置
            TrajectoryPoint final_point;
            final_point.position = target_pos_;
            final_point.velocity = Eigen::Vector3d::Zero();
            final_point.acceleration = Eigen::Vector3d::Zero();
            final_point.yaw = trajectory_.back().yaw;
            final_point.time = trajectory_.back().time + dt_;
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
    
    double calculateOptimalTimeWithMaxVel(double distance, double limited_max_vel)
    {
        // 使用限制的最大速度计算时间
        double acc_time = limited_max_vel / max_acc_;
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
            return 2 * acc_time + const_vel_dist / limited_max_vel;
        }
    }
    
    void generateHorizontalTrajectory(const Eigen::Vector3d& start, 
                                      const Eigen::Vector3d& target, 
                                      double distance)
    {
        if (distance < 1e-3) return;
        
        Eigen::Vector3d direction = (target - start).normalized();
        double total_time = calculateOptimalTime(distance);
        
        // 获取当前轨迹的时间偏移
        double time_offset = trajectory_.empty() ? 0.0 : trajectory_.back().time + dt_;
        
        // 生成轨迹点
        int num_points = static_cast<int>(total_time / dt_) + 1;
        
        for (int i = 0; i < num_points; ++i)
        {
            double t = i * dt_;
            if (t > total_time) t = total_time;
            
            TrajectoryPoint point;
            
            // 使用梯形速度曲线计算位置、速度、加速度
            calculateTrajectoryPoint(t, total_time, distance, direction, start, point);
            point.time += time_offset;
            
            trajectory_.push_back(point);
        }
        
        // 计算水平移动轨迹的偏航角
        updateTrajectoryYaw();
    }
    
    void generateVerticalTrajectory(const Eigen::Vector3d& start, 
                                    const Eigen::Vector3d& target, 
                                    double distance)
    {
        if (distance < 1e-3) return;
        
        Eigen::Vector3d direction = (target - start).normalized();
        
        // 对于垂直移动，限制最大速度为0.5m/s
        double vertical_max_vel = 0.5;
        double total_time = calculateOptimalTimeWithMaxVel(distance, vertical_max_vel);
        
        // 获取当前轨迹的时间偏移
        double time_offset = trajectory_.empty() ? 0.0 : trajectory_.back().time + dt_;
        
        // 生成轨迹点
        int num_points = static_cast<int>(total_time / dt_) + 1;
        
        for (int i = 0; i < num_points; ++i)
        {
            double t = i * dt_;
            if (t > total_time) t = total_time;
            
            TrajectoryPoint point;
            
            // 使用梯形速度曲线计算位置、速度、加速度（使用限制的最大速度）
            calculateTrajectoryPointWithMaxVel(t, total_time, distance, direction, start, point, vertical_max_vel);
            point.time += time_offset;
            
            // 垂直移动时平滑地将偏航角归零
            double initial_yaw = trajectory_.empty() ? last_yaw_ : trajectory_.back().yaw;
            double target_yaw = 0.0;
            double yaw_progress = t / total_time;  // 从0到1的进度
            
            // 使用余弦插值实现更平滑的偏航角过渡
            double smooth_progress = (1.0 - cos(yaw_progress * M_PI)) * 0.5;
            point.yaw = initial_yaw * (1.0 - smooth_progress) + target_yaw * smooth_progress;
            point.yaw = normalizeAngle(point.yaw);
            
            trajectory_.push_back(point);
        }
    }
    
    void updateTrajectoryYaw()
    {
        if (trajectory_.empty()) return;
        
        // 计算水平移动段的偏航角
        size_t start_index = 0;
        
        // 找到当前段的起始位置
        for (size_t i = 1; i < trajectory_.size(); ++i)
        {
            if (std::abs(trajectory_[i].position.z() - trajectory_[i-1].position.z()) > 1e-3)
            {
                start_index = i;
                break;
            }
        }
        
        double yaw_prev = last_yaw_;
        for (size_t i = start_index; i < trajectory_.size(); ++i)
        {
            const auto& pt = trajectory_[i];
            double vx = pt.velocity.x();
            double vy = pt.velocity.y();
            double yaw = yaw_prev;

            // 优先使用速度方向
            if (std::hypot(vx, vy) > 1e-3)
            {
                yaw = std::atan2(vy, vx);
            }
            else if (i + 1 < trajectory_.size())
            {
                // 使用相邻点的位移方向
                Eigen::Vector3d d = trajectory_[i + 1].position - pt.position;
                if (std::hypot(d.x(), d.y()) > 1e-3)
                {
                    yaw = std::atan2(d.y(), d.x());
                }
            }

            yaw = normalizeAngle(yaw);
            trajectory_[i].yaw = yaw;
            yaw_prev = yaw;
        }
    }
    
    void calculateTrajectoryPoint(double t, double total_time, double total_distance, 
                                  const Eigen::Vector3d& direction, const Eigen::Vector3d& start_pos,
                                  TrajectoryPoint& point)
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
        point.position = start_pos + direction * s;
        point.velocity = direction * std::max(0.0, v);
        point.acceleration = direction * a;
        // yaw 在生成轨迹后统一计算
        point.yaw = last_yaw_;
        point.time = t;
    }
    
    void calculateTrajectoryPointWithMaxVel(double t, double total_time, double total_distance, 
                                            const Eigen::Vector3d& direction, const Eigen::Vector3d& start_pos,
                                            TrajectoryPoint& point, double limited_max_vel)
    {
        double acc_time = limited_max_vel / max_acc_;
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
                s = acc_dist + limited_max_vel * (t - acc_time);
                v = limited_max_vel;
                a = 0.0;
            }
            else
            {
                double dt_from_decel = t - (total_time - acc_time);
                s = total_distance - acc_dist + limited_max_vel * dt_from_decel - 0.5 * max_acc_ * dt_from_decel * dt_from_decel;
                v = limited_max_vel - max_acc_ * dt_from_decel;
                a = -max_acc_;
            }
        }
        
        // 转换为3D向量
        point.position = start_pos + direction * s;
        point.velocity = direction * std::max(0.0, v);
        point.acceleration = direction * a;
        // yaw 在生成轨迹后统一计算
        point.yaw = last_yaw_;
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

        // 记录已发布的偏航角
        last_yaw_ = point.yaw;
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
        
        // 停止时偏航角归零，与垂直移动的最终状态保持一致
        cmd.yaw = 0.0;
        cmd.yaw_dot = 0.0;
        
        pos_cmd_pub_.publish(cmd);
        
        ROS_INFO("[SimpleEgoPlanner] Published stop command with yaw: 0.0 rad (0.0 deg)");
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
