// This is the Lidar Odometry And Mapping (LOAM) for solid-state lidar (for example: livox lidar),
// which suffer form motion blur due the continously scan pattern and low range of fov.

// Developer: Jiarong Lin  ziv.lin.ljr@gmail.com

//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// livox点云特征提取
// 对livox点云，按照不同的瓣进行划分, 并提取平面点和角点, 发给后续节点

#ifndef LIVOX_LASER_SCAN_HANDLER_HPP
#define LIVOX_LASER_SCAN_HANDLER_HPP

#include <cmath>
#include <vector>

#define USE_HASH 1
#define SHOW_OPENCV_VIS 0
#if USE_HASH
#include <unordered_map>
#endif

#include <Eigen/Eigen>
#include <Eigen/Eigen>
#include <nav_msgs/Odometry.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include "tools/common.h"
#include "tools/pcl_tools.hpp"
#include "tools/tools_eigen_math.hpp"
#include "tools/tools_logger.hpp"
#define PCL_DATA_SAVE_DIR "/home/ziv/data/loam_pc"

#define IF_LIVOX_HANDLER_REMOVE 0
#define IF_APPEND 0
#define printf_line printf( " %s %d \r\n", __FILE__, __LINE__ );

using namespace std;
using namespace PCL_TOOLS;

class Livox_laser
{
  public:
    string SOFT_WARE_VERSION = string( "V_0.1_beta" );

    enum E_point_type
    {
        e_pt_normal = 0,                      // normal points
        e_pt_000 = 0x0001 << 0,               // points [0,0,0]
        e_pt_too_near = 0x0001 << 1,          // points in short range
        e_pt_reflectivity_low = 0x0001 << 2,  // low reflectivity
        e_pt_reflectivity_high = 0x0001 << 3, // high reflectivity
        e_pt_circle_edge = 0x0001 << 4,       // points near the edge of circle
        e_pt_nan = 0x0001 << 5,               // points with infinite value
        e_pt_small_view_angle = 0x0001 << 6,  // points with large viewed angle
    };

    enum E_feature_type // if and only if normal point can be labeled
    {
        e_label_invalid = -1,
        e_label_unlabeled = 0,
        e_label_corner = 0x0001 << 0,
        e_label_surface = 0x0001 << 1,
        e_label_near_nan = 0x0001 << 2,
        e_label_near_zero = 0x0001 << 3,
        e_label_hight_intensity = 0x0001 << 4,
    };

    // Encode point infos using points intensity, which is more convenient for debugging.
    enum E_intensity_type
    {
        e_I_raw = 0,
        e_I_motion_blur,
        e_I_motion_mix,
        e_I_sigma,
        e_I_scan_angle,
        e_I_curvature,
        e_I_view_angle,
        e_I_time_stamp
    };

    struct Pt_infos    // 存储点的附加信息
    {
        int   pt_type = e_pt_normal;
        int   pt_label = e_label_unlabeled;
        int   idx = 0.f;
        float raw_intensity = 0.f;
        float time_stamp = 0.0;
        float polar_angle = 0.f;
        int   polar_direction = 0;
        float polar_dis_sq2 = 0.f;
        float depth_sq2 = 0.f;
        float curvature = 0.0;
        float view_angle = 0.0;
        float sigma = 0.0;
        Eigen::Matrix< float, 2, 1 > pt_2d_img; // project to X==1 plane
    };

    // E_intensity_type   default_return_intensity_type = e_I_raw;
    E_intensity_type default_return_intensity_type = e_I_motion_blur;
    // E_intensity_type default_return_intensity_type = e_I_scan_angle;
    // E_intensity_type default_return_intensity_type = e_I_curvature;
    // E_intensity_type default_return_intensity_type = e_I_view_angle;

    int   pcl_data_save_index = 0;

    float max_fov = 17; // Edge of circle to main axis
    float m_max_edge_polar_pos = 0;
    float m_time_internal_pts = 1.0e-5; // 10us = 1e-5
    float m_cx = 0;
    float m_cy = 0;
    int   m_if_save_pcd_file = 0;
    int   m_input_points_size;
    double m_first_receive_time = -1;
    double m_current_time;
    double m_last_maximum_time_stamp;
    float thr_corner_curvature = 0.05;
    float thr_surface_curvature = 0.01;
    float minimum_view_angle = 10;
    std::vector< Pt_infos >  m_pts_info_vec;
    std::vector< PointType > m_raw_pts_vec;
#if USE_HASH
    std::unordered_map< PointType, Pt_infos *, Pt_hasher, Pt_compare >           m_map_pt_idx; // using hash_map
    std::unordered_map< PointType, Pt_infos *, Pt_hasher, Pt_compare >::iterator m_map_pt_idx_it;
#else
    std::map< PointType, Pt_infos *, Pt_compare >           m_map_pt_idx;
    std::map< PointType, Pt_infos *, Pt_compare >::iterator m_map_pt_idx_it;
#endif

    float m_livox_min_allow_dis = 1.0;
    float m_livox_min_sigma = 7e-3;

    std::vector< pcl::PointCloud< pcl::PointXYZI > > m_last_laser_scan;

    int     m_img_width = 800;
    int     m_img_heigh = 800;
    ADD_SCREEN_PRINTF_OUT_METHOD;
    ~Livox_laser() {}

    Livox_laser()
    {
        // Some data init
        ENABLE_SCREEN_PRINTF;
        screen_out << "========= Hello, this is livox laser ========" << endl;
        screen_out << "Compile time:  " << __TIME__ << endl;
        screen_out << "Softward version: " << SOFT_WARE_VERSION << endl;
        screen_out << "========= End ========" << endl;

        m_max_edge_polar_pos = std::pow( tan( max_fov / 57.3 ) * 1, 2 );
    }

    template < typename T >
    T dis2_xy( T x, T y )
    {
        return x * x + y * y;
    }

    template < typename T >
    T depth2_xyz( T x, T y, T z )
    {
        return x * x + y * y + z * z;
    }

    template < typename T >
    T depth_xyz( T x, T y, T z )
    {
        return sqrt( depth2_xyz( x, y, z ) );
    }

    template < typename T >
    Pt_infos *find_pt_info(const T & pt )
    {
        m_map_pt_idx_it = m_map_pt_idx.find( pt );    // 点的附加信息vector, <点云, 附加信息>
        //printf( "Input pt is [%lf, %lf, %lf]\r\n", pt.x, pt.y, pt.z );
        if ( m_map_pt_idx_it == m_map_pt_idx.end() )
        {
            printf( "Input pt is [%lf, %lf, %lf]\r\n", pt.x, pt.y, pt.z );
            printf( "Error!!!!\r\n" );
            assert( m_map_pt_idx_it != m_map_pt_idx.end() ); // else, there must be something error happened before.
        }
        return m_map_pt_idx_it->second;  // 返回点云的附加信息
    }

    void get_features( pcl::PointCloud< PointType > &pc_corners, pcl::PointCloud< PointType > &pc_surface, pcl::PointCloud< PointType > &pc_full_res, float minimum_blur = 0.0, float maximum_blur = 0.3 )
    {
        int corner_num = 0;
        int surface_num = 0;
        int full_num = 0;
        pc_corners.resize( m_pts_info_vec.size() );
        pc_surface.resize( m_pts_info_vec.size() );
        pc_full_res.resize( m_pts_info_vec.size() );
        float maximum_idx = maximum_blur * m_pts_info_vec.size();
        float minimum_idx = minimum_blur * m_pts_info_vec.size();
        int pt_critical_rm_mask = e_pt_000 | e_pt_nan | e_pt_too_near;
        for ( size_t i = 0; i < m_pts_info_vec.size(); i++ )
        {
            if ( m_pts_info_vec[ i ].idx > maximum_idx ||
                 m_pts_info_vec[ i ].idx < minimum_idx )
                // 该点的索引和设置的最大最小索引比较 超出了则不考虑该点
                continue;

            if ( ( m_pts_info_vec[ i ].pt_type & pt_critical_rm_mask ) == 0 )
            {
                // 点不为 0 0 0点 或者 nan 点
                if ( m_pts_info_vec[ i ].pt_label & e_label_corner )
                {
                    if ( m_pts_info_vec[ i ].pt_type != e_pt_normal )
                        continue;

                    // 点为 corner 点 且点不属于比较垃圾的点
                    if ( m_pts_info_vec[ i ].depth_sq2 < std::pow( 30, 2 )  )
                    {
                        // 点深度小于30才加入 corner point ??? 为蛤
                        pc_corners.points[ corner_num ] = m_raw_pts_vec[ i ];
                        //set_intensity( pc_corners.points[ corner_num ], e_I_motion_blur );
                        pc_corners.points[ corner_num ].intensity = m_pts_info_vec[ i ].time_stamp;
                        corner_num++;
                    }
                }
                if ( m_pts_info_vec[ i ].pt_label & e_label_surface )
                {
                    if ( m_pts_info_vec[ i ].depth_sq2 < std::pow( 1000, 2 ) )
                    {
                        // 点加入surface points
                        pc_surface.points[ surface_num ] = m_raw_pts_vec[ i ];
                        pc_surface.points[ surface_num ].intensity = float(m_pts_info_vec[ i ].time_stamp);
                        //set_intensity( pc_surface.points[ surface_num ], e_I_motion_blur );
                        surface_num++;
                    }
                }


            }
            pc_full_res.points[ full_num ] = m_raw_pts_vec[ i ];
            pc_full_res.points[ full_num ].intensity = m_pts_info_vec[ i ].time_stamp;
            full_num++; // 点全部加入full集合
            }
        }

        //printf("Get_features , corner num = %d, suface num = %d, blur from %.2f~%.2f\r\n", corner_num, surface_num, minimum_blur, maximum_blur);
        pc_corners.resize(corner_num);
        pc_surface.resize(surface_num);
        pc_full_res.resize(full_num);
    }

    template < typename T >
    void set_intensity( T &pt, const E_intensity_type &i_type = e_I_motion_blur )
    // 设置输出点的intensity通道内容
    {
        Pt_infos *pt_info = find_pt_info( pt );   // 获取该点的附加信息
        switch ( i_type )
        {
        case ( e_I_raw ):
            pt.intensity = pt_info->raw_intensity;
            break;
        case ( e_I_motion_blur ):
            // 默认为此通道 (当前点索引 除去总点数) 代表了该点在该帧点云中的先后位置
            pt.intensity = ( ( float ) pt_info->idx ) / ( float ) m_input_points_size;
            assert( pt.intensity <= 1.0 && pt.intensity >= 0.0 );  // 断言
            break;
        case ( e_I_motion_mix ):
            pt.intensity = 0.1 * ( ( float ) pt_info->idx + 1 ) / ( float ) m_input_points_size + ( int ) ( pt_info->raw_intensity );
            break;
        case ( e_I_scan_angle ):
            pt.intensity = pt_info->polar_angle;
            break;
        case ( e_I_curvature ):
            pt.intensity = pt_info->curvature;
            break;
        case ( e_I_view_angle ):
            pt.intensity = pt_info->view_angle;
            break;
        case (e_I_time_stamp):
            pt.intensity = pt_info->time_stamp;
        default:
            pt.intensity = ( ( float ) pt_info->idx + 1 ) / ( float ) m_input_points_size;
        }
        return;
    }

    template < typename T >
    cv::Mat draw_dbg_img( cv::Mat &img, std::vector< T > &pt_list_eigen, cv::Scalar color = cv::Scalar::all( 255 ), int radius = 3 )
    {
        cv::Mat      res_img = img.clone();
        unsigned int pt_size = pt_list_eigen.size();

        for ( unsigned int idx = 0; idx < pt_size; idx++ )
        {
            draw_pt( res_img, pt_list_eigen[ idx ], color, radius );
        }

        return res_img;
    }

    void add_mask_of_point( Pt_infos *pt_infos, const E_point_type &pt_type, int neighbor_count = 0 )
    {

        int idx = pt_infos->idx;
        pt_infos->pt_type |= pt_type;

        if ( neighbor_count > 0 )
        {
            for ( int i = -neighbor_count; i < neighbor_count; i++ )
            {
                idx = pt_infos->idx + i;

                if ( i != 0 && ( idx >= 0 ) && ( idx < ( int ) m_pts_info_vec.size() ) )
                {
                    //cout << "Add mask, id  = " << idx << "  type = " << pt_type << endl;
                    m_pts_info_vec[ idx ].pt_type |= pt_type;
                }
            }
        }
    }

    void eval_point( Pt_infos *pt_info )
    {
        if ( pt_info->depth_sq2 < m_livox_min_allow_dis * m_livox_min_allow_dis ) // to close
        {
            //cout << "Add mask, id  = " << idx << "  type = e_too_near" << endl;
            add_mask_of_point( pt_info, e_pt_too_near );
        }

        pt_info->sigma = pt_info->raw_intensity / pt_info->polar_dis_sq2;

        if ( pt_info->sigma < m_livox_min_sigma )
        {
            //cout << "Add mask, id  = " << idx << "  type = e_reflectivity_low" << endl;
            add_mask_of_point( pt_info, e_pt_reflectivity_low );
        }
    }

    // compute curvature and view angle
    void compute_features()
    {
        unsigned int pts_size = m_raw_pts_vec.size();
        size_t       curvature_ssd_size = 2;
        int          critical_rm_point = e_pt_000 | e_pt_nan;
        float        neighbor_accumulate_xyz[ 3 ] = { 0.0, 0.0, 0.0 };

        //cout << "Surface_thr = " << thr_surface_curvature << " , corner_thr = " << thr_corner_curvature<< " ,minimum_view_angle = " << minimum_view_angle << endl;
        for ( size_t idx = curvature_ssd_size; idx < pts_size - curvature_ssd_size; idx++ )
        {
            if ( m_pts_info_vec[ idx ].pt_type & critical_rm_point )
            {
                // 若该点为000点或者nan点 不处理
                continue;
            }

            /*********** Compute curvate ************/
            neighbor_accumulate_xyz[ 0 ] = 0.0;
            neighbor_accumulate_xyz[ 1 ] = 0.0;
            neighbor_accumulate_xyz[ 2 ] = 0.0;

            for ( size_t i = 1; i <= curvature_ssd_size; i++ )
            {
                if ( ( m_pts_info_vec[ idx + i ].pt_type & e_pt_000 ) || ( m_pts_info_vec[ idx - i ].pt_type & e_pt_000 ) )
                {
                    //该点的前后点 若为000点 则退出 并记录该点invalid
                    if ( i == 1 )
                    {
                        m_pts_info_vec[ idx ].pt_label |= e_label_near_zero;
                    }
                    else
                    {
                        m_pts_info_vec[ idx ].pt_label = e_label_invalid;
                    }
                    break;
                }
                else if ( ( m_pts_info_vec[ idx + i ].pt_type & e_pt_nan ) || ( m_pts_info_vec[ idx - i ].pt_type & e_pt_nan ) )
                {
                    //该点的前后点 若为nan点 则退出 并记录该点invalid
                    if ( i == 1 )
                    {
                        m_pts_info_vec[ idx ].pt_label |= e_label_near_nan;
                    }
                    else
                    {
                        m_pts_info_vec[ idx ].pt_label = e_label_invalid;
                    }
                    break;
                }
                else
                {
                    // 该点左右各 curvature_ssd_size个点, 把这 2 * curvature_ssd_size 个点的x y z坐标加起来
                    neighbor_accumulate_xyz[ 0 ] += m_raw_pts_vec[ idx + i ].x + m_raw_pts_vec[ idx - i ].x;
                    neighbor_accumulate_xyz[ 1 ] += m_raw_pts_vec[ idx + i ].y + m_raw_pts_vec[ idx - i ].y;
                    neighbor_accumulate_xyz[ 2 ] += m_raw_pts_vec[ idx + i ].z + m_raw_pts_vec[ idx - i ].z;
                }
            }

            if(m_pts_info_vec[ idx ].pt_label == e_label_invalid)
            {
                //若该点周围有点为 nan或0 0 0 则不进行处理
                continue;
            }

            // x_cur = (x_-2 + x_-1 + x_0 + x_1 + x_2) - 4 * x_0
            // y_cur = (y_-2 + y_-1 + y_0 + y_1 + y_2) - 4 * y_0
            // z_cur = (z_-2 + z_-1 + z_0 + z_1 + z_2) - 4 * z_0
            neighbor_accumulate_xyz[ 0 ] -= curvature_ssd_size * 2 * m_raw_pts_vec[ idx ].x;
            neighbor_accumulate_xyz[ 1 ] -= curvature_ssd_size * 2 * m_raw_pts_vec[ idx ].y;
            neighbor_accumulate_xyz[ 2 ] -= curvature_ssd_size * 2 * m_raw_pts_vec[ idx ].z;
            m_pts_info_vec[ idx ].curvature = neighbor_accumulate_xyz[ 0 ] * neighbor_accumulate_xyz[ 0 ] + neighbor_accumulate_xyz[ 1 ] * neighbor_accumulate_xyz[ 1 ] +
                                              neighbor_accumulate_xyz[ 2 ] * neighbor_accumulate_xyz[ 2 ];
            // m_pts_info_vec[idx] = x_cur^2 + y_cur^2 + z_cur^2 该点曲率计算

            /*********** Compute plane angle ************/
            Eigen::Matrix< float, 3, 1 > vec_a( m_raw_pts_vec[ idx ].x, m_raw_pts_vec[ idx ].y, m_raw_pts_vec[ idx ].z );
            // 该点的 xyz
            Eigen::Matrix< float, 3, 1 > vec_b( m_raw_pts_vec[ idx + curvature_ssd_size ].x - m_raw_pts_vec[ idx - curvature_ssd_size ].x,
                                                m_raw_pts_vec[ idx + curvature_ssd_size ].y - m_raw_pts_vec[ idx - curvature_ssd_size ].y,
                                                m_raw_pts_vec[ idx + curvature_ssd_size ].z - m_raw_pts_vec[ idx - curvature_ssd_size ].z );
            // 该点前面两点与该点后面两点的连线 组成的直线

            m_pts_info_vec[ idx ].view_angle = Eigen_math::vector_angle( vec_a  , vec_b, 1 ) * 57.3;   // 求出该点到原点的线，以及该点前后两点的连线的夹角

            if ( m_pts_info_vec[ idx ].view_angle > minimum_view_angle ) // 夹角至少10°
            {

                if( m_pts_info_vec[ idx ].curvature < thr_surface_curvature )  // 曲率小于thr_surface_curvature
                {
                    m_pts_info_vec[ idx ].pt_label |= e_label_surface;  // 设置该点为平面点
                }

                float sq2_diff = 0.1;

                if ( m_pts_info_vec[ idx ].curvature > thr_corner_curvature ) // 曲率大 为角点
                {
                    if ( m_pts_info_vec[ idx ].depth_sq2 <= m_pts_info_vec[ idx - curvature_ssd_size ].depth_sq2 &&
                         m_pts_info_vec[ idx ].depth_sq2 <= m_pts_info_vec[ idx + curvature_ssd_size ].depth_sq2 )
                    // 中间点 (当前点) 的深度 <= 该点两侧点的深度
                    {
                        if ( abs( m_pts_info_vec[ idx ].depth_sq2 - m_pts_info_vec[ idx - curvature_ssd_size ].depth_sq2 ) < sq2_diff * m_pts_info_vec[ idx ].depth_sq2 ||
                             abs( m_pts_info_vec[ idx ].depth_sq2 - m_pts_info_vec[ idx + curvature_ssd_size ].depth_sq2 ) < sq2_diff * m_pts_info_vec[ idx ].depth_sq2 )
                            // [abs(当前点的深度 - 左侧两点深度) < 0.1 * 当前点深度] 或 [abs(当前点的深度 - 右侧两点深度) < 0.1 * 当前点深度]
                            m_pts_info_vec[ idx ].pt_label |= e_label_corner;
                            // 设置该点为角点
                    }
                }
            }
        }
    }

    template < typename T >
    int projection_scan_3d_2d( pcl::PointCloud< T > &laserCloudIn, std::vector< float > &scan_id_index )
    {

        unsigned int pts_size = laserCloudIn.size();
        m_pts_info_vec.clear();
        m_pts_info_vec.resize( pts_size );
        m_raw_pts_vec.resize( pts_size );
        std::vector< int > edge_idx;
        std::vector< int > split_idx;
        scan_id_index.resize( pts_size );
        m_map_pt_idx.clear();
        m_map_pt_idx.reserve( pts_size );
        std::vector< int > zero_idx;

        m_input_points_size = 0;

        for ( unsigned int idx = 0; idx < pts_size; idx++ )
        {
            //遍历每个点
            m_raw_pts_vec[ idx ] = laserCloudIn.points[ idx ];    // 点原始信息
            Pt_infos *pt_info = &m_pts_info_vec[ idx ];           // 点附加信息
            m_map_pt_idx.insert( std::make_pair( laserCloudIn.points[ idx ], pt_info ) );   // 原始点与附加信息合成pair
            pt_info->raw_intensity = laserCloudIn.points[ idx ].intensity;
            pt_info->idx = idx;
            pt_info->time_stamp = m_current_time + ( ( float ) idx ) * m_time_internal_pts;
            // 保存点的原始强度、第几个索引、相对起始时间戳
            m_last_maximum_time_stamp = pt_info->time_stamp; // 保存当前帧最后一个点的时间戳
            m_input_points_size++;

            if ( !std::isfinite( laserCloudIn.points[ idx ].x ) ||
                 !std::isfinite( laserCloudIn.points[ idx ].y ) ||
                 !std::isfinite( laserCloudIn.points[ idx ].z ) )
            {
                add_mask_of_point( pt_info, e_pt_nan ); //该点有坐标是infinite, 则设置该点类型为infinite
                continue;
            }

            if ( laserCloudIn.points[ idx ].x == 0 )   //该点坐标为[0 0 0]
            {
                if ( idx == 0 )
                {
                    // TODO: handle this case.
                    screen_out << "First point should be normal!!!" << std::endl;

                    pt_info->pt_2d_img << 0.01, 0.01;
                    pt_info->polar_dis_sq2 = 0.0001;
                    add_mask_of_point( pt_info, e_pt_000 );
                    //return 0;
                }
                else
                {
                    //若该点不是该帧的第一个点, 该点的pt_2d_img, polar_dis_sq2和上一个点保持一致
                    pt_info->pt_2d_img = m_pts_info_vec[ idx - 1 ].pt_2d_img;
                    pt_info->polar_dis_sq2 = m_pts_info_vec[ idx - 1 ].polar_dis_sq2;
                    add_mask_of_point( pt_info, e_pt_000 );  // 设置该点类型为[0 0 0]点
                    continue;
                }
            }
            // 若点为0 0 0或者infinite, 则不进行此处后续计算

            m_map_pt_idx.insert( std::make_pair( laserCloudIn.points[ idx ], pt_info ) );   // 原始点与附加信息合成pair

            pt_info->depth_sq2 = depth2_xyz( laserCloudIn.points[ idx ].x, laserCloudIn.points[ idx ].y, laserCloudIn.points[ idx ].z );
            // 点range^2

            pt_info->pt_2d_img << laserCloudIn.points[ idx ].y / laserCloudIn.points[ idx ].x, laserCloudIn.points[ idx ].z / laserCloudIn.points[ idx ].x;
            pt_info->polar_dis_sq2 = dis2_xy( pt_info->pt_2d_img( 0 ), pt_info->pt_2d_img( 1 ) );
            // 点pt_2d_img = y/x, z/x
            // 点polar_dis_sq2 = (y^2+z^2)/x^2

            eval_point( pt_info ); // 判断点是否属于超近点、或者强度超低点 

            if ( pt_info->polar_dis_sq2 > m_max_edge_polar_pos ) //大于视场角
            {
                add_mask_of_point( pt_info, e_pt_circle_edge, 2 ); // 判断点属于投影边缘点
            }

            // 前面都是对点进行分类及计算点的部分参数
            // Split scans
            if ( idx >= 1 )
            {
                //判断点是靠近花瓣中心or远离花瓣中心
                float dis_incre = pt_info->polar_dis_sq2 - m_pts_info_vec[ idx - 1 ].polar_dis_sq2;

                if ( dis_incre > 0 ) // far away from zero
                {
                    pt_info->polar_direction = 1;          // 当前点的角度大于上一个点角度 远离花瓣中心
                }

                if ( dis_incre < 0 ) // move toward zero   // 当前点的角度小于上一个点角度 靠近花瓣中心
                {
                    pt_info->polar_direction = -1;
                }

                if ( pt_info->polar_direction == -1 && m_pts_info_vec[ idx - 1 ].polar_direction == 1 )
                {
                    // 上一个点远离 这个点靠近 说明是花瓣顶点
                    if ( edge_idx.size() == 0 || ( idx - split_idx[ split_idx.size() - 1 ] ) > 50 )
                    {
                        // 距离加入上一个点比较远了 则加入新的花瓣顶点
                        split_idx.push_back( idx );
                        edge_idx.push_back( idx );
                        continue;
                    }
                }

                if ( pt_info->polar_direction == 1 && m_pts_info_vec[ idx - 1 ].polar_direction == -1 )
                {
                    // 上一个点靠近 这个点远离 说明是花瓣中心点
                    if ( zero_idx.size() == 0 || ( idx - split_idx[ split_idx.size() - 1 ] ) > 50 )
                    {
                        // 距离加入上一个点比较远了 则加入新的花瓣零点
                        split_idx.push_back( idx );
                        zero_idx.push_back( idx );
                        continue;
                    }
                }
            }
        }

        split_idx.push_back( pts_size - 1 );  // 加入最后一个点

        int   val_index = 0;
        int   pt_angle_index = 0;
        float scan_angle = 0;
        int   internal_size = 0;

        if( split_idx.size() < 6) // minimum 3 petal of scan.
            return 0;
        
        for ( int idx = 0; idx < ( int ) pts_size; idx++ )
        {
            // idx == 0: 第一个点
            // idx > split_idx[ val_index + 1 ]: 下一个转折点或者0点
            if ( val_index < split_idx.size() - 2 )
            {
                if ( idx == 0 || idx > split_idx[ val_index + 1 ] )
                {
                    if ( idx > split_idx[ val_index + 1 ] )
                    {
                        val_index++;
                         // 到转折点, 则 val_index + 1
                    }

                    internal_size = split_idx[ val_index + 1 ] - split_idx[ val_index ];
                    // 当前这个花瓣顶点或花瓣中心点, 到下一个花瓣顶点或中心点之间的点数目

                    if ( m_pts_info_vec[ split_idx[ val_index + 1 ] ].polar_dis_sq2 > 10000 )
                    // 若下一个点的 (y^2+z^2)/x^2 > 10000, 下一个点是个距离原点很近的点???
                    {
                        pt_angle_index = split_idx[ val_index + 1 ] - ( int ) ( internal_size * 0.20 );
                        //选取距离当前点4/5花瓣的点, 即距离下个点1/5花瓣的点, 计算该点相对花瓣中心的角度情况
                        scan_angle = atan2( m_pts_info_vec[ pt_angle_index ].pt_2d_img( 1 ), m_pts_info_vec[ pt_angle_index ].pt_2d_img( 0 ) ) * 57.3;
                        scan_angle = scan_angle + 180.0;
                    }
                    else
                    {
                        //选取距离当前点1/5花瓣的点, 即距离下个点4/5花瓣的点, 计算该点相对花瓣中心的角度情况
                        pt_angle_index = split_idx[ val_index + 1 ] - ( int ) ( internal_size * 0.80 );
                        scan_angle = atan2( m_pts_info_vec[ pt_angle_index ].pt_2d_img( 1 ), m_pts_info_vec[ pt_angle_index ].pt_2d_img( 0 ) ) * 57.3;
                        scan_angle = scan_angle + 180.0;
                    }

                    //cout << "Idx  = " << idx <<  " val = "<< val_index << "  angle = " << scan_angle << endl;
                }
            }
            m_pts_info_vec[ idx ].polar_angle = scan_angle; // 该点在投影圆面上的角度
            scan_id_index[ idx ] = scan_angle; // 该点在投影圆面上的角度
        }

        return split_idx.size() - 1;  // 返回花瓣数目
    }

    // will be delete...
    //    template<typename T>
    void reorder_laser_cloud_scan( std::vector< pcl::PointCloud< pcl::PointXYZI > > &in_laserCloudScans, std::vector< std::vector< int > > &pts_mask )
    {
        unsigned int min_pts_per_scan = 0;
        screen_out << "Before reorder" << endl;
        //cout << "Cloud size: " << in_laserCloudScans.size() << endl;
        //std::vector< pcl::PointCloud< PointType > > res_laser_cloud( in_laserCloudScans.size() - 2 ); // abandon first and last
        //std::vector<std::vector<int>> res_pts_mask( in_laserCloudScans.size() - 2 );
        std::vector< pcl::PointCloud< pcl::PointXYZI > > res_laser_cloud( in_laserCloudScans.size() ); // abandon first and last
        std::vector< std::vector< int > >                res_pts_mask( in_laserCloudScans.size() );
        std::map< float, int > map_angle_idx;

        // for (unsigned int i = 1; i < in_laserCloudScans.size() - 1; i++ )
        for ( unsigned int i = 0; i < in_laserCloudScans.size() - 0; i++ )
        {
            if ( in_laserCloudScans[ i ].size() > min_pts_per_scan )
            {
                //cout << i << endl;
                //cout << "[" << i << "] size = ";
                //cout << in_laserCloudScans[ i ].size() << "  ,id = " << ( int ) in_laserCloudScans[ i ].points[ 0 ].intensity << endl;
                map_angle_idx.insert( std::make_pair( in_laserCloudScans[ i ].points[ 0 ].intensity, i ) );
            }
            else
            {
                continue;
            }
        }

        screen_out << "After reorder" << endl;
        std::map< float, int >::iterator it;
        int current_index = 0;

        for ( it = map_angle_idx.begin(); it != map_angle_idx.end(); it++ )
        {
            //cout << "[" << current_index << "] id = " << it->first << endl;
            if ( in_laserCloudScans[ it->second ].size() > min_pts_per_scan )
            {
                res_laser_cloud[ current_index ] = in_laserCloudScans[ it->second ];
                res_pts_mask[ current_index ] = pts_mask[ it->second ];
                current_index++;
            }
        }

        res_laser_cloud.resize( current_index );
        res_pts_mask.resize( current_index );

        in_laserCloudScans = res_laser_cloud;
        pts_mask = res_pts_mask;
        screen_out << "Return size = " << pts_mask.size() << "  " << in_laserCloudScans.size() << endl;
        return;
    }

    // Split whole point cloud into scans.
    template < typename T >
    void split_laser_scan( const int clutter_size, const pcl::PointCloud< T > &laserCloudIn,
                           const std::vector< float > &                 scan_id_index,
                           std::vector< pcl::PointCloud< PointType > > &laserCloudScans )
    {
        // 点云划分为scans
        std::vector< std::vector< int > > pts_mask;
        laserCloudScans.resize( clutter_size );  // size 设置为 clutter_size
        pts_mask.resize( clutter_size );         // 
        PointType point;
        int       scan_idx = 0;

        for ( unsigned int i = 0; i < laserCloudIn.size(); i++ )
        {

            point = laserCloudIn.points[ i ];           // 当前点
            
            if ( i > 0 && ( ( scan_id_index[ i ] ) != ( scan_id_index[ i - 1 ] ) ) )
            {   // i > 0, 且当前点的花瓣角度和上一帧的花瓣角度不一样, 说明是一个新的花瓣
                // std::cout << "Scan idx = " << scan_idx << " intensity = " << scan_id_index[ i ] << std::endl;
                scan_idx = scan_idx + 1;
                pts_mask[scan_idx].reserve( 5000 );    // 预留该花瓣的存储空间
            }
            
            laserCloudScans[ scan_idx ].push_back( point );                  // 向该花瓣的存储空间插入新点
            pts_mask[ scan_idx ].push_back( m_pts_info_vec[ i ].pt_type );   // 向该花瓣存入该点的点类型信息
        }
        laserCloudScans.resize(scan_idx);   // 这是神马操作? 


        int remove_point_pt_type = e_pt_000 |
                                   e_pt_too_near |
                                   e_pt_nan  
                                //    e_pt_circle_edge
                                   ;
        int scan_avail_num = 0;
        std::vector< pcl::PointCloud< PointType > > res_laser_cloud_scan;
        for ( unsigned int i = 0; i < laserCloudScans.size(); i++ )
        {
            scan_avail_num = 0;
            pcl::PointCloud< PointType > laser_clour_per_scan;
            // 对每个花瓣进行遍历

            for ( unsigned int idx = 0; idx < laserCloudScans[ i ].size(); idx++ )
            // 对该花瓣内的点进行遍历
            {
                if ( ( pts_mask[ i ][ idx ] & remove_point_pt_type ) == 0 ) // 若该点不是应删除的点
                {
                    if ( laserCloudScans[ i ].points[ idx ].x == 0 )   // 若该点为0 0 0, 重新计算下个点 (且说明前面处理有问题) 不应该有此类点
                    {
                        screen_printf( "Error!!! Mask = %d\r\n", pts_mask[ i ][ idx ] );
                        assert( laserCloudScans[ i ].points[ idx ].x != 0 );
                        continue;
                    }
                    auto temp_pt = laserCloudScans[ i ].points[ idx ];
                    set_intensity( temp_pt, default_return_intensity_type );   // 设置点的intensity通道
                    laser_clour_per_scan.points.push_back(temp_pt);               // 插入新点
                    scan_avail_num++;
                }
            }

            //printf(" %d|%d number of point in this scan  = %d ------------------\r\n ", i, laserCloudScans.size(), scan_avail_num);
            if(scan_avail_num)
            {
                res_laser_cloud_scan.push_back(laser_clour_per_scan);
            }
        }
        laserCloudScans= res_laser_cloud_scan;
    }

    template < typename T >
    std::vector< pcl::PointCloud< pcl::PointXYZI > > extract_laser_features( pcl::PointCloud< T > &laserCloudIn, double time_stamp = -1 )
    {
        assert(time_stamp >= 0.0);
        if(time_stamp <= 0.0000001 || (time_stamp < m_last_maximum_time_stamp) ) // old firmware, without timestamp
        {
            m_current_time = m_last_maximum_time_stamp;
        }
        else
        {
          m_current_time = time_stamp - m_first_receive_time;
        }
        if ( m_first_receive_time <= 0 )
        {
            m_first_receive_time = time_stamp;
        }

        screen_out << "Extract feature, input: " << time_stamp << ", first time: " << m_first_receive_time << ", current_time: " << m_current_time << endl;
        std::vector< pcl::PointCloud< PointType > > laserCloudScans, temp_laser_scans;
        std::vector< float >                        scan_id_index;
        laserCloudScans.clear();
        m_map_pt_idx.clear();
        m_pts_info_vec.clear();
        m_raw_pts_vec.clear();
        //printf_line;
        if ( m_if_save_pcd_file )
        {
            stringstream ss;
            ss << PCL_DATA_SAVE_DIR << "/pc_" << pcl_data_save_index << ".pcd" << endl;
            pcl_data_save_index = pcl_data_save_index + 1;
            screen_out << "Save file = " << ss.str() << endl;
            pcl::io::savePCDFileASCII( ss.str(), laserCloudIn );
        }

        int clutter_size = projection_scan_3d_2d( laserCloudIn, scan_id_index );  // 将点划分到2D平面上, 即sqrt(y^2+z^2)和x的平面上, 就是将花瓣投影在正对激光雷达的面上
        // clutter_size = 一帧数据里花瓣的数量
        // scan_id_index: 保存每个点在投影平面圆上的角度 注意(一个花瓣内的角度是一致的)

        compute_features();    // 计算每个点的曲率和视角, 以及该点是平面点还是角点
        if ( clutter_size == 0 )
        {
            // 没有花瓣
            return laserCloudScans;
        }
        else
        {
            // 有花瓣 将点云按不同花瓣进行划分
            // clutter_size:    花瓣数量
            // laserCloudIn:    输入点云
            // scan_id_index:   点在圆面上的投影角度
            // laserCloudScans: 保存不同花瓣点的vector
            split_laser_scan( clutter_size, laserCloudIn, scan_id_index, laserCloudScans );
            return laserCloudScans;
        }
    }
};

#endif // LIVOX_LASER_SCAN_HANDLER_HPP
