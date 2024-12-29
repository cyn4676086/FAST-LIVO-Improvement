#include "lidar_selection.h"
#include "camera_manager.h"
#include <iterator>
#include <vector>
namespace lidar_selection {

LidarSelector::LidarSelector(int gridsize, SparseMap* sparsemap, std::vector<camera_manager::Cameras> &cameras_info)
    : grid_size(gridsize), sparse_map(sparsemap), state(nullptr), state_propagat(nullptr),
      Rli(M3D::Identity()), Pli(V3D::Zero()),
      sub_sparse_map(nullptr),
      ncc_en(false), debug(0), patch_size(0), patch_size_total(0), patch_size_half(0),
      MIN_IMG_COUNT(0), NUM_MAX_ITERATIONS(0), weight_function_(nullptr),
      weight_scale_(1.0f), img_point_cov(1.0), outlier_threshold(1.0), ncc_thre(0.5),
      n_meas_(0), computeH(0.0), ekf_time(0.0),
      ave_total(0.0), frame_count(0),
      scale_estimator_(nullptr)
{
    
   
    downSizeFilter.setLeafSize(0.2f, 0.2f, 0.2f);
    G = Matrix<double, DIM_STATE, DIM_STATE>::Zero();
    H_T_H = Matrix<double, DIM_STATE, DIM_STATE>::Zero();
    
    // 多相机初始化
    for(const auto& cam_info_struct : cameras_info) {
        CameraInfo cam_info;
        cam_info.cam = cam_info_struct.cam;
        cam_info.cam_id = cam_info_struct.cam_id;

        // 内参
        cam_info.fx = cam_info_struct.cam_fx;
        cam_info.fy = cam_info_struct.cam_fy;
        cam_info.cx = cam_info_struct.cam_cx;
        cam_info.cy = cam_info_struct.cam_cy;

        // 图像尺寸
        cam_info.width = cam_info_struct.cam_width;
        cam_info.height = cam_info_struct.cam_height;

        // 外参
        cam_info.Rcl = cam_info_struct.Rcl;
        cam_info.Pcl = cam_info_struct.Pcl;

        cam_info.Rci = cam_info.Rcl * Rli;
        cam_info.Pci = cam_info.Rcl * Pli + cam_info.Pcl;
        //初始化为0 在状态更新中计算
        cam_info.Rcw = M3D::Identity();
        cam_info.Pcw = V3D::Zero();

        // 雅可比矩阵
        cam_info.Jdphi_dR = cam_info.Rci;
        cam_info.Jdp_dR = -cam_info.Rci * (SKEW_SYM_MATRX(-cam_info.Rci.transpose() * cam_info.Pci));
        cam_info.Jdp_dt = V3D::Identity();
        // 添加到相机列表
        cameras.push_back(cam_info);
    }
}
LidarSelector::~LidarSelector() 
{
    delete[] align_flag;
    delete[] grid_num;
    delete[] map_index;
    delete[] map_value;

    delete sparse_map;
    delete sub_sparse_map;

    unordered_map<int, Warp*>().swap(Warp_map);
    unordered_map<VOXEL_KEY, float>().swap(sub_feat_map);
    unordered_map<VOXEL_KEY, VOXEL_POINTS*>().swap(feat_map);  
}
void LidarSelector::init()
{
    //图像尺寸一致 建图参数一致
    width = cameras[0].width;
    height = cameras[0].height;
    sub_sparse_map = new SubSparseMap;
    grid_n_width = static_cast<int>(width/grid_size);
    grid_n_height = static_cast<int>(height/grid_size);
    length = grid_n_width * grid_n_height;
    grid_num = new int[length];
    map_index = new int[length];
    map_value = new float[length];
    map_dist = (float*)malloc(sizeof(float)*length);

    // 使用 std::fill_n 替代 memset
    std::fill_n(grid_num, length, TYPE_UNKNOWN);
    std::fill_n(map_index, length, 0);
    std::fill_n(map_value, length, 0.0f);

    // memset(grid_num, TYPE_UNKNOWN, sizeof(int)*length);
    // memset(map_index, 0, sizeof(int)*length);
    // memset(map_value, 0, sizeof(float)*length);
    voxel_points_.reserve(length);
    add_voxel_points_.reserve(length);
    patch_size_total = patch_size * patch_size;
    patch_size_half = static_cast<int>(patch_size/2);
    patch_cache.resize(patch_size_total);
    stage_ = STAGE_FIRST_FRAME;
    pg_down.reset(new PointCloudXYZI());

    // Initialize weight functions
    weight_scale_ = 10.0f;
    weight_function_.reset(new vk::robust_cost::HuberWeightFunction());
    // weight_function_.reset(new vk::robust_cost::TukeyWeightFunction());
    scale_estimator_.reset(new vk::robust_cost::UnitScaleEstimator());
    // scale_estimator_.reset(new vk::robust_cost::MADScaleEstimator());
}


//雷达和IMU外参 保持不变
void LidarSelector::set_extrinsic(const V3D &transl, const M3D &rot)
{
    Pli = -rot.transpose() * transl;
    Rli = rot.transpose();
}

void LidarSelector::reset_grid()
{
    memset(grid_num, TYPE_UNKNOWN, sizeof(int)*length);
    memset(map_index, 0, sizeof(int)*length);
    fill_n(map_dist, length, 10000);
    std::vector<PointPtr>(length).swap(voxel_points_);
    std::vector<V3D>(length).swap(add_voxel_points_);
    voxel_points_.reserve(length);
    add_voxel_points_.reserve(length);
}

void LidarSelector::dpi(const V3D& p, MD(2,3)& J, double fx, double fy) {
    const double x = p[0];
    const double y = p[1];
    const double z_inv = 1.0 / p[2];
    const double z_inv_2 = z_inv * z_inv;
    
    J(0,0) = fx * z_inv;
    J(0,1) = 0.0;
    J(0,2) = -fx * x * z_inv_2;
    
    J(1,0) = 0.0;
    J(1,1) = fy * z_inv;
    J(1,2) = -fy * y * z_inv_2;
}


//区分多相机
float LidarSelector::CheckGoodPoints(const cv::Mat& img, const Eigen::Vector2d& uv, int cam_id)
{
    CameraInfo& cam_info = cameras[cam_id];
    const float u_ref = uv[0];
    const float v_ref = uv[1];
    const int u_ref_i = floorf(u_ref); 
    const int v_ref_i = floorf(v_ref);
    if(u_ref_i < 1 || u_ref_i >= cam_info.width - 1 || 
       v_ref_i < 1 || v_ref_i >= cam_info.height - 1)
        return 0.0f;

    // 计算梯度
    float gu = 0.0f;
    float gv = 0.0f;
    for(int dy = -1; dy <=1; dy++)
    {
        for(int dx = -1; dx <=1; dx++)
        {
            if(dy == 0 && dx == 0) continue;
            gu += (img.at<uint8_t>(v_ref_i + dy, u_ref_i + dx) - img.at<uint8_t>(v_ref_i + dy, u_ref_i - dx)) * (dx != 0 ? 2.0f : 1.0f);
            gv += (img.at<uint8_t>(v_ref_i + dy, u_ref_i + dx) - img.at<uint8_t>(v_ref_i - dy, u_ref_i + dx)) * (dy != 0 ? 2.0f : 1.0f);
        }
    }
    return fabs(gu) + fabs(gv);
}



void LidarSelector::getpatch(const cv::Mat& img, const Eigen::Vector2d& pc, float* patch_tmp, int level, int cam_id) 
{
    CameraInfo& cam_info = cameras[cam_id];
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int scale =  (1 << level);
    const int u_ref_i = floorf(u_ref / scale) * scale; 
    const int v_ref_i = floorf(v_ref / scale) * scale;
    const float subpix_u_ref = (u_ref - u_ref_i) / scale;
    const float subpix_v_ref = (v_ref - v_ref_i) / scale;
    const float w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0f - subpix_v_ref);
    const float w_ref_bl = (1.0f - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;

    for (int x = 0; x < patch_size; x++) 
    {
        int row = v_ref_i - patch_size_half * scale + x * scale;
        if(row < 0 || row >= cam_info.height - scale) continue;
        uint8_t* img_ptr = img.data + row * cam_info.width + (u_ref_i - patch_size_half * scale);
        for (int y = 0; y < patch_size; y++, img_ptr += scale)
        {
            int col = u_ref_i - patch_size_half * scale + y * scale;
            if(col < 0 || col >= cam_info.width - scale) 
            {
                patch_tmp[patch_size_total * level + x * patch_size + y] = 0.0f;
                continue;
            }
            patch_tmp[patch_size_total * level + x * patch_size + y] = 
                w_ref_tl * img_ptr[0] + 
                w_ref_tr * img_ptr[scale] + 
                w_ref_bl * img_ptr[scale * cam_info.width] + 
                w_ref_br * img_ptr[scale * cam_info.width + scale];
        }
    }
}


// 添加稀疏地图
void LidarSelector::addSparseMap(const cv::Mat& img, PointCloudXYZI::Ptr pg, int cam_id) 
{
    CameraInfo& cam_info = cameras[cam_id];

    // 重置网格数据
    std::fill_n(grid_num, grid_n_width * grid_n_height, TYPE_UNKNOWN);
    std::fill_n(map_index, grid_n_width * grid_n_height, 0);
    std::fill_n(map_value, grid_n_width * grid_n_height, 0.0f);
    std::fill_n(map_dist, grid_n_width * grid_n_height, 10000.0f);

    for (int i = 0; i < pg->size(); i++) 
    {
        Eigen::Vector3d pt(pg->points[i].x, pg->points[i].y, pg->points[i].z);
        Eigen::Vector2d pc = new_frame_->w2c(pt, cam_id);
        if(cam_info.cam->isInFrame(pc.cast<int>(), (patch_size_half + 1) * 8)) // 20px is the patch size in the matcher
        {
            int index = static_cast<int>(pc[0] / grid_size) * grid_n_height + static_cast<int>(pc[1] / grid_size);
            float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);

            if (cur_value > map_value[index]) // && (grid_num[index] != TYPE_MAP || map_value[index]<=10)) //! only add in not occupied grid
            {
                map_value[index] = cur_value;
                add_voxel_points_[index] = pt;
                grid_num[index] = TYPE_POINTCLOUD;
            }
        }
    }

    int add = 0;
    for (int i = 0; i < length; i++) 
    {
        if (grid_num[i] == TYPE_POINTCLOUD) // && (map_value[i]>=10)) //! debug
        {
            Eigen::Vector3d pt = add_voxel_points_[i];
            Eigen::Vector2d pc = new_frame_->w2c(pt, cam_id);

            PointPtr pt_new(new Point(pt));
            Eigen::Vector3d f = cameras[cam_id].cam->cam2world(pc);
            FeaturePtr ftr_new(new Feature(pc, f, new_frame_->T_f_w_, map_value[i], 0, cam_id));
            ftr_new->img = new_frame_->img_pyr_[cam_id][0];
            ftr_new->id_ = new_frame_->id_;

            pt_new->addFrameRef(ftr_new);
            pt_new->value = map_value[i];
            AddPoint(pt_new);
            add += 1;
        }
    }
    printf("[ VIO ][Camera %d]: Add %d 3D points.\n", cam_id, add);
}

void LidarSelector::AddPoint(PointPtr pt_new)
{
    V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
    double voxel_size = 0.5;
    float loc_xyz[3];
    for(int j=0; j<3; j++)
    {
      loc_xyz[j] = pt_w[j] / voxel_size;
      if(loc_xyz[j] < 0)
      {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_KEY position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter != feat_map.end())
    {
      iter->second->voxel_points.push_back(pt_new);
      iter->second->count++;
    }
    else
    {
      VOXEL_POINTS *ot = new VOXEL_POINTS(0);
      ot->voxel_points.push_back(pt_new);
      feat_map[position] = ot;
    }
}

void LidarSelector::getWarpMatrixAffine(
    const vk::AbstractCamera& cam,
    const Vector2d& px_ref,
    const Vector3d& f_ref,
    const double depth_ref,
    const SE3& T_cur_ref,
    const int level_ref,    // the corresponding pyrimid level of px_ref
    const int pyramid_level,
    const int halfpatch_size,
    Matrix2d& A_cur_ref)
{
  // Compute affine warp matrix A_ref_cur
  const Vector3d xyz_ref(f_ref*depth_ref);
  Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size,0)*(1<<level_ref)*(1<<pyramid_level)));
  Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0,halfpatch_size)*(1<<level_ref)*(1<<pyramid_level)));
//   Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size,0)*(1<<level_ref)));
//   Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0,halfpatch_size)*(1<<level_ref)));
  xyz_du_ref *= xyz_ref[2]/xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2]/xyz_dv_ref[2];
  const Vector2d px_cur(cam.world2cam(T_cur_ref*(xyz_ref)));
  const Vector2d px_du(cam.world2cam(T_cur_ref*(xyz_du_ref)));
  const Vector2d px_dv(cam.world2cam(T_cur_ref*(xyz_dv_ref)));
  A_cur_ref.col(0) = (px_du - px_cur)/halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur)/halfpatch_size;
}

// 仿射变换
void LidarSelector::warpAffine(const Eigen::Matrix2d& A_cur_ref, const cv::Mat& img_ref, const Eigen::Vector2d& px_ref,
                               int level_ref, int search_level, int pyramid_level, int halfpatch_size,
                               float* patch, int cam_id)
{
    const int patch_size = halfpatch_size * 2;
    const Eigen::Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
    if(std::isnan(A_ref_cur(0,0)))
    {
        printf("Affine warp is NaN, probably camera has no translation\n"); 
        return;
    }

    for(int y = 0; y < patch_size; ++y)
    {
        for(int x = 0; x < patch_size; ++x)
        {
            Eigen::Vector2f px_patch(x - halfpatch_size, y - halfpatch_size);
            px_patch *= (1 << search_level);
            px_patch *= (1 << pyramid_level);
            Eigen::Vector2f px = A_ref_cur * px_patch + px_ref.cast<float>();
            if(px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1)
                patch[patch_size_total * pyramid_level + y * patch_size + x] = 0.0f;
            else
                patch[patch_size_total * pyramid_level + y * patch_size + x] = 
                    static_cast<float>(vk::interpolateMat_8u(img_ref, px[0], px[1]));
        }
    }
}

double LidarSelector::NCC(float* ref_patch, float* cur_patch, int patch_size)
{    
    double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
    double mean_ref =  sum_ref / patch_size;

    double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
    double mean_curr =  sum_cur / patch_size;

    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < patch_size; i++) 
    {
        double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
        numerator += n;
        demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);
        demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

int LidarSelector::getBestSearchLevel(
    const Matrix2d& A_cur_ref,
    const int max_level)
{
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();

  while(D > 3.0 && search_level < max_level)
  {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

#ifdef FeatureAlign
void LidarSelector::createPatchFromPatchWithBorder(float* patch_with_border, float* patch_ref)
{
  float* ref_patch_ptr = patch_ref;
  for(int y=1; y<patch_size+1; ++y, ref_patch_ptr += patch_size)
  {
    float* ref_patch_border_ptr = patch_with_border + y*(patch_size+2) + 1;
    for(int x=0; x<patch_size; ++x)
      ref_patch_ptr[x] = ref_patch_border_ptr[x];
  }
}
#endif

void LidarSelector::addFromSparseMap(const std::vector<cv::Mat>& imgs, PointCloudXYZI::Ptr pg)
{
    if(feat_map.empty()) return;

    // 确保传入的图像数量与相机数量匹配
    if(imgs.size() != cameras.size()) {
        std::cerr << "[ERROR] Number of images does not match number of cameras." << std::endl;
        return;
    }

    // 下采样点云
    pg_down->clear();
    downSizeFilter.setInputCloud(pg);
    downSizeFilter.filter(*pg_down);
    
    // 重置网格数据
    std::fill_n(grid_num, grid_n_width * grid_n_height, TYPE_UNKNOWN);
    std::fill_n(map_index, grid_n_width * grid_n_height, 0);
    std::fill_n(map_value, grid_n_width * grid_n_height, 0.0f);
    std::fill_n(map_dist, grid_n_width * grid_n_height, 10000.0f);

    sub_sparse_map->reset();
    sub_map_cur_frame_.clear();

    float voxel_size = 0.5f;

    // 清空子特征图和 Warp 映射
    sub_feat_map.clear();
    for(auto& kv : Warp_map) {
        delete kv.second;
    }
    Warp_map.clear();

    // 为每个相机创建一个深度图
    std::vector<cv::Mat> depth_imgs(cameras.size());
    std::vector<float*> depth_ptrs(cameras.size());

    for(size_t cam_idx = 0; cam_idx < cameras.size(); ++cam_idx) {
        CameraInfo& cam_info = cameras[cam_idx];
        depth_imgs[cam_idx] = cv::Mat::zeros(cam_info.height, cam_info.width, CV_32FC1);
        depth_ptrs[cam_idx] = reinterpret_cast<float*>(depth_imgs[cam_idx].data);
    }

    // 遍历下采样后的点云
    for(int i = 0; i < static_cast<int>(pg_down->size()); i++)
    {
        // 转换点到世界坐标系
        Eigen::Vector3d pt_w(pg_down->points[i].x, pg_down->points[i].y, pg_down->points[i].z);

        // 计算体素键值
        int loc_xyz[3];
        for(int j = 0; j < 3; j++)
        {
            loc_xyz[j] = static_cast<int>(std::floor(pt_w[j] / voxel_size));
        }
        VOXEL_KEY position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

        // 更新子特征图
        sub_feat_map[position] = 1.0f;

        // 遍历每个相机，更新深度图
        for(size_t cam_idx = 0; cam_idx < cameras.size(); ++cam_idx)
        {
            CameraInfo& cam_info = cameras[cam_idx];
            Eigen::Vector3d pt_cam = new_frame_->w2f(pt_w); 

            if(pt_cam[2] > 0.0)
            {
                // 像素坐标
                float u = cam_info.fx * static_cast<float>(pt_cam[0] / pt_cam[2]) + cam_info.cx;
                float v = cam_info.fy * static_cast<float>(pt_cam[1] / pt_cam[2]) + cam_info.cy;

                int col = static_cast<int>(u);
                int row = static_cast<int>(v);

                // 若在图像内，则更新 depth
                if(col >= 0 && col < cam_info.width && row >= 0 && row < cam_info.height)
                {
                    float& current_depth = depth_ptrs[cam_idx][cam_info.width * row + col];
                    float depth_cam = static_cast<float>(pt_cam[2]);
                    if(current_depth == 0.0f || depth_cam < current_depth)
                    {
                        current_depth = depth_cam;
                    }
                }
            }
        }
    }

    // 多相机并行处理
    #pragma omp parallel for
    for(int cam_idx = 0; cam_idx < static_cast<int>(cameras.size()); cam_idx++)
    {
        CameraInfo& cam_info = cameras[cam_idx];

        // 转灰度
        cv::Mat img_mono;
        if(imgs[cam_idx].channels() > 1)
            cv::cvtColor(imgs[cam_idx], img_mono, cv::COLOR_BGR2GRAY);
        else
            img_mono = imgs[cam_idx].clone();

        // 根据深度图生成单相机对应的稀疏点云 pg_cam
        PointCloudXYZI::Ptr pg_cam(new PointCloudXYZI());
        pg_cam->reserve(static_cast<size_t>(cam_info.width * cam_info.height));

        float* depth_ptr = depth_ptrs[cam_idx];

        for(int row = 0; row < cam_info.height; row++)
        {
            for(int col = 0; col < cam_info.width; col++)
            {
                float depth = depth_ptr[row * cam_info.width + col];
                if(depth > 0.0f && depth < 10000.0f) // 确保深度有效
                {
                    // 反投影得到3D点
                    float x = (static_cast<float>(col) - cam_info.cx) * depth / cam_info.fx;
                    float y = (static_cast<float>(row) - cam_info.cy) * depth / cam_info.fy;
                    float z = depth;

                    // 无法使用大括号初始化(四个float)，需要先临时对象赋值
                    pcl::PointXYZINormal p;
                    p.x = x;
                    p.y = y;
                    p.z = z;
                    p.intensity = 1.0f;

                    pg_cam->points.push_back(p);
                }
            }
        }

        // 添加稀疏点云
        addSparseMap(img_mono, pg_cam, cam_idx);
    }
}


#ifdef FeatureAlign
bool LidarSelector::align2D(
    const cv::Mat& cur_img,
    float* ref_patch_with_border,
    float* ref_patch,
    const int n_iter,
    Vector2d& cur_px_estimate,
    int index)
{
#ifdef __ARM_NEON__
  if(!no_simd)
    return align2D_NEON(cur_img, ref_patch_with_border, ref_patch, n_iter, cur_px_estimate);
#endif

  const int halfpatch_size_ = 4;
  const int patch_size_ = 8;
  const int patch_area_ = 64;
  bool converged=false;

  // compute derivative of template and prepare inverse compositional
  float __attribute__((__aligned__(16))) ref_patch_dx[patch_area_];
  float __attribute__((__aligned__(16))) ref_patch_dy[patch_area_];
  Matrix3f H; H.setZero();

  // compute gradient and hessian
  const int ref_step = patch_size_+2;
  float* it_dx = ref_patch_dx;
  float* it_dy = ref_patch_dy;
  for(int y=0; y<patch_size_; ++y) 
  {
    float* it = ref_patch_with_border + (y+1)*ref_step + 1; 
    for(int x=0; x<patch_size_; ++x, ++it, ++it_dx, ++it_dy)
    {
      Vector3f J;
      J[0] = 0.5 * (it[1] - it[-1]); 
      J[1] = 0.5 * (it[ref_step] - it[-ref_step]); 
      J[2] = 1; 
      *it_dx = J[0];
      *it_dy = J[1];
      H += J*J.transpose(); 
    }
  }
  Matrix3f Hinv = H.inverse();
  float mean_diff = 0;

  // Compute pixel location in new image:
  float u = cur_px_estimate.x();
  float v = cur_px_estimate.y();

  // termination condition
  const float min_update_squared = 0.03*0.03;//0.03*0.03
  const int cur_step = cur_img.step.p[0];
  float chi2 = 0;
  chi2 = sub_sparse_map->propa_errors[index];
  Vector3f update; update.setZero();
  for(int iter = 0; iter<n_iter; ++iter)
  {
    int u_r = floor(u);
    int v_r = floor(v);
    if(u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols-halfpatch_size_ || v_r >= cur_img.rows-halfpatch_size_)
      break;

    if(isnan(u) || isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
      return false;

    // compute interpolation weights
    float subpix_x = u-u_r;
    float subpix_y = v-v_r;
    float wTL = (1.0-subpix_x)*(1.0-subpix_y);
    float wTR = subpix_x * (1.0-subpix_y);
    float wBL = (1.0-subpix_x)*subpix_y;
    float wBR = subpix_x * subpix_y;

    // loop through search_patch, interpolate
    float* it_ref = ref_patch;
    float* it_ref_dx = ref_patch_dx;
    float* it_ref_dy = ref_patch_dy;
    float new_chi2 = 0.0;
    Vector3f Jres; Jres.setZero();
    for(int y=0; y<patch_size_; ++y)
    {
      uint8_t* it = (uint8_t*) cur_img.data + (v_r+y-halfpatch_size_)*cur_step + u_r-halfpatch_size_; 
      for(int x=0; x<patch_size_; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy)
      {
        float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[cur_step] + wBR*it[cur_step+1];
        float res = search_pixel - *it_ref + mean_diff;
        Jres[0] -= res*(*it_ref_dx);
        Jres[1] -= res*(*it_ref_dy);
        Jres[2] -= res;
        new_chi2 += res*res;
      }
    }

    if(iter > 0 && new_chi2 > chi2)
    {
    //   cout << "error increased." << endl;
      u -= update[0];
      v -= update[1];
      break;
    }
    chi2 = new_chi2;

    sub_sparse_map->align_errors[index] = new_chi2;

    update = Hinv * Jres;
    u += update[0];
    v += update[1];
    mean_diff += update[2];

#if SUBPIX_VERBOSE
    cout << "Iter " << iter << ":"
         << "\t u=" << u << ", v=" << v
         << "\t update = " << update[0] << ", " << update[1]
//         << "\t new chi2 = " << new_chi2 << endl;
#endif

    if(update[0]*update[0]+update[1]*update[1] < min_update_squared)
    {
#if SUBPIX_VERBOSE
      cout << "converged." << endl;
#endif
      converged=true;
      break;
    }
  }

  cur_px_estimate << u, v;
  return converged;
}

void LidarSelector::FeatureAlignment(cv::Mat img)
{
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return;
    memset(align_flag, 0, length);
    int FeatureAlignmentNum = 0;
       
    for (int i=0; i<total_points; i++) 
    {
        bool res;
        int search_level = sub_sparse_map->search_levels[i];
        Vector2d px_scaled(sub_sparse_map->px_cur[i]/(1<<search_level));
        res = align2D(new_frame_->img_pyr_[search_level], sub_sparse_map->patch_with_border[i], sub_sparse_map->patch[i],
                        20, px_scaled, i);
        sub_sparse_map->px_cur[i] = px_scaled * (1<<search_level);
        if(res)
        {
            align_flag[i] = 1;
            FeatureAlignmentNum++;
        }
    }
}
#endif


float LidarSelector::UpdateState(const std::vector<cv::Mat>& imgs, float total_residual, int level) 
{
    int total_points = sub_sparse_map->index.size();
    if (total_points == 0) return 0.0f;

    // 保存旧状态，以便在优化失败时恢复
    StatesGroup old_state = (*state);
    bool EKF_end = false;

    // 初始化误差和测量计数
    float error = 0.0f;
    float last_error = total_residual;
    n_meas_ = 0;

    // 准备测量向量 z 和雅可比矩阵 H_sub
    // 假设每个点在每个相机中都有一个测量，H_DIM 为总测量数
    // H_sub 的列数为状态维度，这里假设状态维度为 DIM_STATE
    // H_sub 的行数为 H_DIM
    const int H_DIM_PER_CAM = total_points * patch_size_total;
    const int H_DIM = H_DIM_PER_CAM * cameras.size();
    Eigen::VectorXd z(H_DIM);
    z.setZero();

    // 重置 H_sub 矩阵
    H_sub.resize(H_DIM, DIM_STATE);
    H_sub.setZero();

    // 获取当前状态的旋转和平移
    M3D Rwi(state->rot_end); // 世界到IMU的旋转矩阵
    V3D Pwi(state->pos_end); // 世界到IMU的平移向量

    // 计算相机到世界的变换
    // 对每个相机分别计算 Rcw 和 Pcw
    for(size_t cam_idx = 0; cam_idx < cameras.size(); cam_idx++)
    {
        CameraInfo& cam_info = cameras[cam_idx];
        cam_info.Rcw = cam_info.Rci * Rwi.transpose();
        cam_info.Pcw = -cam_info.Rci * Rwi.transpose() * Pwi + cam_info.Pci;
    }

    // 开始迭代优化
    for (int iteration = 0; iteration < NUM_MAX_ITERATIONS; iteration++) 
    {
        error = 0.0f;
        n_meas_ = 0;

        // 遍历每个相机
        for(size_t cam_idx = 0; cam_idx < cameras.size(); cam_idx++)
        {
            CameraInfo& cam_info = cameras[cam_idx];
            const cv::Mat& img_gray = imgs[cam_idx];

            // 遍历每个点
            for(int i = 0; i < total_points; i++)
            {
                PointPtr pt = sub_sparse_map->voxel_points[i];
                if(pt == nullptr) continue;

                // 变换点到相机坐标系
                Eigen::Vector3d pf = cam_info.Rcw * pt->pos_ + cam_info.Pcw;

                // 点在相机前方
                if(pf[2] <= 0) continue;

                // 投影到图像平面
                Eigen::Vector2d pc = cameras[cam_idx].cam->world2cam(pf);
                if(!cameras[cam_idx].cam->isInFrame(pc.cast<int>(), (patch_size_half + 1) * 8))
                    continue;

                // 计算雅可比矩阵
                Eigen::Matrix<double, 2, 3> Jdpi;
                dpi(pf, Jdpi, cam_info.fx, cam_info.fy); // 计算投影的雅可比

                // 计算 SKEW_SYM_MATRX(pf)
                Eigen::Matrix<double, 3, 3> p_hat;
                p_hat << 
                    0, -pf[2], pf[1],
                    pf[2], 0, -pf[0],
                    -pf[1], pf[0], 0;

                // 提取补丁
                float patch_ref[patch_size_total];
                getpatch(img_gray, pc, patch_ref, level, cam_idx);

                int search_level = sub_sparse_map->search_levels[i];
                int pyramid_level = level + search_level;
                const int scale =  (1<<pyramid_level);
                const int u_ref_i = floorf(pc[0]/scale)*scale; 
                const int v_ref_i = floorf(pc[1]/scale)*scale;
                // 计算残差和雅可比
                for(int x = 0; x < patch_size; x++) 
                {
                    for(int y = 0; y < patch_size; y++) 
                    {
                        int idx = cam_idx * H_DIM_PER_CAM + i * patch_size_total + x * patch_size + y;

                        // 获取图像像素值
                        float img_val = static_cast<float>(img_gray.at<uchar>(v_ref_i + x * scale - patch_size_half * scale, 
                                                                             u_ref_i + y * scale - patch_size_half * scale));

                        // 计算残差
                        float res = img_val - patch_ref[x * patch_size + y];
                        z(idx) = res;
                        error += res * res;
                        n_meas_++;

                        // 计算图像梯度 du, dv
                        float du = 0.5f * ((img_gray.at<uchar>(v_ref_i + x * scale - patch_size_half * scale, u_ref_i + y * scale + scale - patch_size_half * scale) 
                                           - img_gray.at<uchar>(v_ref_i + x * scale - patch_size_half * scale, u_ref_i + y * scale - scale - patch_size_half * scale)));
                        float dv = 0.5f * ((img_gray.at<uchar>(v_ref_i + x * scale + scale - patch_size_half * scale, u_ref_i + y * scale - patch_size_half * scale) 
                                           - img_gray.at<uchar>(v_ref_i + x * scale - scale - patch_size_half * scale, u_ref_i + y * scale - patch_size_half * scale)));

                        Eigen::Vector2d Jimg(du, dv);
                        Jimg = Jimg * (1.0 / scale);

                        // 计算雅可比
                        Eigen::Vector3d Jdphi = Jimg.transpose() * Jdpi;
                        Eigen::Vector3d Jdp = -Jimg.transpose() * Jdpi;

                        // 计算旋转和平移的雅可比
                        Eigen::Matrix<double, 1, DIM_STATE> JH;
                        JH.setZero();
                        // 假设前3维为旋转，后3维为平移
                        JH.segment<3>(0) = Jdphi.transpose();
                        JH.segment<3>(3) = Jdp.transpose();

                        // 填充 H_sub 矩阵
                        H_sub.row(idx) = JH;
                    }
                }
            }
        }

        // 计算平均误差
        error /= static_cast<float>(n_meas_);

        // 检查误差是否有所降低
        if(error <= last_error)
        {
            old_state = (*state);
            last_error = error;

            // 计算 H^T * H 和 H^T * z
            Eigen::MatrixXd Ht = H_sub.transpose();
            H_T_H = Ht * H_sub;

            // 计算卡尔曼增益 K
            Eigen::MatrixXd K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();

            // 计算 H^T * z
            Eigen::VectorXd HTz = Ht * z;

            // 计算状态更新
            Eigen::VectorXd solution = -K_1 * HTz;

            // 更新状态
            (*state) += solution;

            // 检查收敛条件
            if(solution.norm() < 1e-3) // 具体阈值可根据需求调整
            {
                EKF_end = true;
            }
        }
        else
        {
            // 优化未收敛，恢复旧状态
            (*state) = old_state;
            EKF_end = true;
        }

        // 检查是否结束优化
        if(iteration == NUM_MAX_ITERATIONS || EKF_end) 
        {
            break;
        }
    }

    return last_error;
}

//T_f_w_ 应仅基于帧的全局位姿状态 (由IMU提供），不应包含任何相机的外参。
void LidarSelector::updateFrameState(const StatesGroup& state)
{
    M3D Rwi(state.rot_end); // 从状态中获取旋转矩阵
    V3D Pwi(state.pos_end); // 从状态中获取平移向量

    // 更新帧的位姿，仅基于IMU状态，不依赖于相机外参
    new_frame_->T_f_w_ = SE3(Rwi, Pwi);
}

void LidarSelector::addObservation(const std::vector<cv::Mat>& imgs)
{
    int total_points = sub_sparse_map->index.size();
    if (total_points == 0) return;

    for(int i = 0; i < total_points; i++) 
    {
        PointPtr pt = sub_sparse_map->voxel_points[i];
        if(pt == nullptr) continue;

        for(int cam_idx = 0; cam_idx < cameras.size(); cam_idx++)
        {
            int cam_id = cam_idx;
            Eigen::Vector2d pc = new_frame_->w2c(pt->pos_, cam_id);
            SE3 pose_cur = new_frame_->T_f_w_;
            bool add_flag = false;

            // 步骤1: 时间条件
            if(pt->obs_.empty()) continue; // 没有观测则跳过
            FeaturePtr last_feature = pt->obs_.back();

            // 步骤2: 姿态变化条件
            SE3 pose_ref = last_feature->T_f_w_;
            SE3 delta_pose = pose_ref * pose_cur.inverse();
            double delta_p = delta_pose.translation().norm();
            double delta_theta = (delta_pose.rotation_matrix().trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (delta_pose.rotation_matrix().trace() - 1));            
            if(delta_p > 0.5 || delta_theta > 10.0) add_flag = true;

            // 步骤3: 像素距离条件
            Eigen::Vector2d last_px = last_feature->px;
            double pixel_dist = (pc - last_px).norm();
            if(pixel_dist > 40.0) add_flag = true;

            // 保持每个点的观测特征数量
            if(pt->obs_.size() >= 20)
            {
                FeaturePtr ref_ftr;
                pt->getFurthestViewObs(new_frame_->pos(), ref_ftr);
                pt->deleteFeatureRef(ref_ftr);
            }

            if(add_flag)
            {
                float val = vk::shiTomasiScore(imgs[cam_id], pc[0], pc[1]);
                Eigen::Vector3d f = cameras[cam_id].cam->cam2world(pc);
                FeaturePtr ftr_new(new Feature(pc, f, new_frame_->T_f_w_, val, sub_sparse_map->search_levels[i], cam_id)); 
                ftr_new->img = new_frame_->img_pyr_[cam_id][0];
                ftr_new->id_ = new_frame_->id_;
                pt->addFrameRef(ftr_new);      
            }
        }
    }
}


void LidarSelector::ComputeJ(const std::vector<cv::Mat>& imgs) 
{
    int total_points = sub_sparse_map->index.size();
    if (total_points == 0) return;
    
    float error = 1e10f;
    float now_error = error;

    // 遍历金字塔的不同层级
    for (int level = 2; level >= 0; level--) 
    {
        now_error = UpdateState(imgs, error, level);
    }

    if (now_error < error)
    {
        state->cov -= G * state->cov;
    }

    updateFrameState(*state);
}


void LidarSelector::display_keypatch(double time)
{
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return;
    for(int i=0; i<total_points; i++)
    {
        PointPtr pt = sub_sparse_map->voxel_points[i];
        V2D pc(new_frame_->w2c(pt->pos_,0));//默认为主相机
        cv::Point2f pf;
        pf = cv::Point2f(pc[0], pc[1]); 
        if (sub_sparse_map->errors[i]<8000) // 5.5
            cv::circle(img_cp, pf, 6, cv::Scalar(0, 255, 0), -1, 8); // Green Sparse Align tracked
        else
            cv::circle(img_cp, pf, 6, cv::Scalar(255, 0, 0), -1, 8); // Blue Sparse Align tracked
    }   
    std::string text = std::to_string(int(1/time))+" HZ";
    cv::Point2f origin;
    origin.x = 20;
    origin.y = 20;
    cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, 8, 0);
}

// 获取像素值
Eigen::Vector3f LidarSelector::getpixel(const cv::Mat& img, const Eigen::Vector2d& pc, int cam_id) 
{
    CameraInfo& cam_info = cameras[cam_id];
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(u_ref); 
    const int v_ref_i = floorf(v_ref);
    const float subpix_u_ref = (u_ref - u_ref_i);
    const float subpix_v_ref = (v_ref - v_ref_i);
    const float w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0f - subpix_v_ref);
    const float w_ref_bl = (1.0f - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;

    if(u_ref_i < 0 || u_ref_i >= cam_info.width - 1 || 
       v_ref_i < 0 || v_ref_i >= cam_info.height - 1)
        return Eigen::Vector3f(0.0f, 0.0f, 0.0f);

    uint8_t* img_ptr = img.data + ((v_ref_i) * cam_info.width + (u_ref_i)) * 3;
    float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[3] + 
              w_ref_bl * img_ptr[cam_info.width * 3] + 
              w_ref_br * img_ptr[cam_info.width * 3 + 3];
    float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[4] + 
              w_ref_bl * img_ptr[cam_info.width * 3 + 1] + 
              w_ref_br * img_ptr[cam_info.width * 3 + 4];
    float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[5] + 
              w_ref_bl * img_ptr[cam_info.width * 3 + 2] + 
              w_ref_br * img_ptr[cam_info.width * 3 + 5];
    return Eigen::Vector3f(B, G, R);
}


void LidarSelector::detect(const std::vector<cv::Mat>& imgs, PointCloudXYZI::Ptr pg) 
{
    // 确保传入的图像数量与相机数量匹配
    if(imgs.size() != cameras.size())
    {
        std::cerr << "Error: Number of images does not match number of cameras." << std::endl;
        return;
    }

    // 初始化新的帧
    std::vector<vk::AbstractCamera*> cam_ptrs;
    cam_ptrs.reserve(cameras.size());
    for(size_t i = 0; i < cameras.size(); i++){
        cam_ptrs.push_back(cameras[i].cam);
    }

    // 再调用Frame
    new_frame_.reset(new Frame(cam_ptrs, imgs));
    updateFrameState(*state);

    if(stage_ == STAGE_FIRST_FRAME && pg->size() > 10)
    {
        new_frame_->setKeyframe();
        stage_ = STAGE_DEFAULT_FRAME;
    }

    double t1 = omp_get_wtime();

    addFromSparseMap(imgs, pg);
    double t4 = omp_get_wtime();
    
    // 准备所有相机的灰度图
    std::vector<cv::Mat> img_grays(cameras.size());
    for(int cam_idx = 0; cam_idx < cameras.size(); cam_idx++)
    {
        if(imgs[cam_idx].channels() > 1)
            cv::cvtColor(imgs[cam_idx], img_grays[cam_idx], cv::COLOR_BGR2GRAY);
        else
            img_grays[cam_idx] = imgs[cam_idx].clone();
    }

    // 使用所有相机的灰度图进行状态计算
    ComputeJ(img_grays);

    double t5 = omp_get_wtime();

    // 使用所有相机的观测进行添加
    addObservation(imgs);

    double t2 = omp_get_wtime();
    frame_count++;
    ave_total = ave_total * (frame_count - 1) / frame_count + (t2 - t1) / frame_count;

    printf("[ VIO ]: time: addFromSparseMap: %.6f addSparseMap: %.6f ComputeJ: %.6f addObservation: %.6f total time: %.6f ave_total: %.6f.\n",
           t4 - t1, t5 - t4, t5 - t4, t2 - t5, t2 - t1, ave_total);

    display_keypatch(t2 - t1);
}



} // namespace lidar_selection