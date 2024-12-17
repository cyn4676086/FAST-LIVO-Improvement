#include "lidar_selection.h"

namespace lidar_selection {

LidarSelector::LidarSelector(const int gridsize, SparseMap* sparsemap, const std::vector<vk::AbstractCamera*>& cameras)
    : grid_size(gridsize), sparse_map(sparsemap), cams(cameras), stage_(STAGE_FIRST_FRAME), frame_count(0), ave_total(0.0)
{
    // 设置下采样滤波器参数
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);

    // 初始化矩阵
    G = Matrix<double, DIM_STATE, DIM_STATE>::Zero();
    H_T_H = Matrix<double, DIM_STATE, DIM_STATE>::Zero();

    // 初始化雷达到IMU的外参
    Rli = M3D::Identity();
    Pli = V3D::Zero();

    // 初始化相机内参
    size_t num_cameras = cams.size();

    fx.resize(num_cameras, 0.0);
    fy.resize(num_cameras, 0.0);
    cx.resize(num_cameras, 0.0);
    cy.resize(num_cameras, 0.0);

    for(size_t i = 0; i < num_cameras; ++i){
        fx[i] = cams[i]->errorMultiplier2();
        fy[i] = cams[i]->errorMultiplier() / (4.0 * fx[i]);
        cx[i] = cams[i]->cx();
        cy[i] = cams[i]->cy();
    }

    // 假设所有相机具有相同的图像尺寸
    width = cams[0]->width();
    height = cams[0]->height();
    grid_n_width = static_cast<int>(width / grid_size);
    grid_n_height = static_cast<int>(height / grid_size);
    length = grid_n_width * grid_n_height;

    // 初始化多相机相关的数据结构
    align_flag.resize(num_cameras);
    grid_num.resize(num_cameras);
    map_index.resize(num_cameras);
    map_value.resize(num_cameras);
    map_dist.resize(num_cameras);
    patch_with_border_.resize(num_cameras);
    patch_cache.resize(num_cameras, std::vector<float>());

    for(size_t i = 0; i < num_cameras; ++i){
        align_flag[i] = std::make_unique<int[]>(length);
        grid_num[i] = std::make_unique<int[]>(length);
        map_index[i] = std::make_unique<int[]>(length);
        map_value[i] = std::make_unique<float[]>(length);
        map_dist[i] = std::make_unique<float[]>(length);
        patch_with_border_[i] = std::make_unique<float[]>((patch_size + 2) * (patch_size + 2)); // 根据需要调整大小

        // 初始化数组
        std::fill_n(align_flag[i].get(), length, 0);
        std::fill_n(grid_num[i].get(), length, TYPE_UNKNOWN);
        std::fill_n(map_index[i].get(), length, 0);
        std::fill_n(map_value[i].get(), length, 0.0f);
        std::fill_n(map_dist[i].get(), length, 10000.0f); // 初始化为一个较大的值
        std::fill_n(patch_with_border_[i].get(), (patch_size + 2) * (patch_size + 2), 0.0f);

        // 初始化 patch_cache
        patch_cache[i].resize(patch_size_total);
    }

    // 预留空间
    voxel_points_.reserve(length);
    add_voxel_points_.reserve(length);

    // 初始化补丁相关参数
    patch_size_total = patch_size * patch_size;
    patch_size_half = static_cast<int>(patch_size / 2);

    // 初始化点云下采样指针
    pg_down.reset(new PointCloudXYZI());

    // 初始化其他成员变量
    weight_scale_ = 10;
    weight_function_ = std::make_unique<vk::robust_cost::HuberWeightFunction>();
    // weight_function_ = std::make_unique<vk::robust_cost::TukeyWeightFunction>();
    scale_estimator_ = std::make_unique<vk::robust_cost::UnitScaleEstimator>();
    // scale_estimator_ = std::make_unique<vk::robust_cost::MADScaleEstimator>();
}

LidarSelector::~LidarSelector() 
{
    // 使用智能指针管理内存，无需手动删除
    delete sparse_map;
    delete sub_sparse_map;

    for(auto& warp_pair : Warp_map){
        delete warp_pair.second;
    }
    Warp_map.clear();

    for(auto& voxel_pair : feat_map){
        delete voxel_pair.second;
    }
    feat_map.clear();

    // 智能指针会自动释放内存，无需清理 align_flag, grid_num 等
}

void LidarSelector::set_extrinsic(const V3D &transl, const M3D &rot)
{
    // 保持雷达到IMU的外参设置
    Pli = -rot.transpose() * transl;
    Rli = rot.transpose();
}

void LidarSelector::init()
{
    sub_sparse_map = new SubSparseMap;
    
    size_t num_cameras = cams.size();
    
    // 初始化每个相机的外参 (通过 SparseMap)
    for(size_t i = 0; i < num_cameras; ++i){
        Rci = sparse_map->Rcl * Rli;
        Pci = sparse_map->Rcl * Pli + sparse_map->Pcl;
        M3D Ric = Rci;
        V3D Pic = -Rci.transpose() * Pci;
        M3D tmp;
        tmp << SKEW_SYM_MATRX(Pic);
        Jdphi_dR = -Rci * tmp;
        Jdp_dR = -Rci * tmp; // 根据需求调整
    }

    // 初始化网格相关参数（已在构造函数中完成）

    voxel_points_.reserve(length);
    add_voxel_points_.reserve(length);
    stage_ = STAGE_FIRST_FRAME;
    pg_down.reset(new PointCloudXYZI());
}

void LidarSelector::reset_grid()
{
    size_t num_cameras = cams.size();
    for(size_t i = 0; i < num_cameras; ++i){
        std::fill_n(grid_num[i].get(), length, TYPE_UNKNOWN);
        std::fill_n(map_index[i].get(), length, 0);
        std::fill_n(map_dist[i].get(), length, 10000.0f);
        std::fill(voxel_points_.begin(), voxel_points_.end(), nullptr);
        std::fill(add_voxel_points_.begin(), add_voxel_points_.end(), V3D::Zero());
    }
}

void LidarSelector::dpi(const V3D& p, MD(2,3)& J, int cam_idx) {
    const double x = p[0];
    const double y = p[1];
    const double z_inv = 1.0 / p[2];
    const double z_inv_2 = z_inv * z_inv;
    // 使用对应相机的 fx 和 fy
    J(0,0) = fx[cam_idx] * z_inv;
    J(0,1) = 0.0;
    J(0,2) = -fx[cam_idx] * x * z_inv_2;
    J(1,0) = 0.0;
    J(1,1) = fy[cam_idx] * z_inv;
    J(1,2) = -fy[cam_idx] * y * z_inv_2;
}

float LidarSelector::CheckGoodPoints(cv::Mat img, V2D uv, int cam_idx)
{
    const float u_ref = uv[0];
    const float v_ref = uv[1];
    const int u_ref_i = floorf(u_ref); 
    const int v_ref_i = floorf(v_ref);
    const float subpix_u_ref = u_ref - u_ref_i;
    const float subpix_v_ref = v_ref - v_ref_i;

    // 确保索引不越界
    if(u_ref_i <= 0 || v_ref_i <= 0 || u_ref_i >= width -1 || v_ref_i >= height -1){
        return 0.0f;
    }

    uint8_t* img_ptr = img.data + (v_ref_i * width + u_ref_i);
    float gu = 2.0f * (img_ptr[1] - img_ptr[-1]) + img_ptr[1 - width] - img_ptr[-1 - width] + img_ptr[1 + width] - img_ptr[-1 + width];
    float gv = 2.0f * (img_ptr[width] - img_ptr[-width]) + img_ptr[width + 1] - img_ptr[-width + 1] + img_ptr[width -1] - img_ptr[-width -1];
    return fabs(gu) + fabs(gv);
}

void LidarSelector::getpatch(cv::Mat img, V2D pc, float* patch_tmp, int level, int cam_idx) 
{
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int scale = (1 << level);
    const int u_ref_i = floorf(pc[0] / scale) * scale; 
    const int v_ref_i = floorf(pc[1] / scale) * scale;
    const float subpix_u_ref = (u_ref - u_ref_i) / scale;
    const float subpix_v_ref = (v_ref - v_ref_i) / scale;
    const float w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0f - subpix_v_ref);
    const float w_ref_bl = (1.0f - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;

    for (int x = 0; x < patch_size; x++) 
    {
        int img_x = u_ref_i - patch_size_half * scale + x * scale;
        int img_y = v_ref_i - patch_size_half * scale;
        
        // 确保索引不越界
        if(img_x < 0 || img_x >= width - scale || img_y < 0 || img_y >= height - scale){
            std::fill(patch_tmp + patch_size_total * level + x * patch_size, patch_tmp + patch_size_total * level + (x +1) * patch_size, 0.0f);
            continue;
        }

        uint8_t* img_ptr = img.data + (img_y * width + img_x);
        for (int y = 0; y < patch_size; y++, img_ptr += scale)
        {
            patch_tmp[patch_size_total * level + x * patch_size + y] = 
                w_ref_tl * img_ptr[0] + 
                w_ref_tr * img_ptr[scale] + 
                w_ref_bl * img_ptr[scale * width] + 
                w_ref_br * img_ptr[scale * width + scale];
        }
    }
}

void LidarSelector::addSparseMap(cv::Mat img, PointCloudXYZI::Ptr pg) 
{
    reset_grid();

    for (int i = 0; i < pg->size(); i++) 
    {
        V3D pt(pg->points[i].x, pg->points[i].y, pg->points[i].z);
        // 处理所有相机
        for(size_t cam_idx = 0; cam_idx < cams.size(); ++cam_idx){
            V2D pc = new_frame_->w2c(pt, cam_idx); // 修改 w2c 以支持多相机
            if(cams[cam_idx]->isInFrame(pc.cast<int>(), (patch_size_half + 1) * 8)) // 20px is the patch size in the matcher
            {
                int grid_x = static_cast<int>(pc[0] / grid_size);
                int grid_y = static_cast<int>(pc[1] / grid_size);
                if(grid_x < 0 || grid_x >= grid_n_width || grid_y < 0 || grid_y >= grid_n_height){
                    continue;
                }
                int index = grid_x * grid_n_height + grid_y;
                float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);

                if (cur_value > map_value[cam_idx][index]) //&& (grid_num[cam_idx][index] != TYPE_MAP || map_value[cam_idx][index] <= 10)) //! only add in not occupied grid
                {
                    map_value[cam_idx][index] = cur_value;
                    add_voxel_points_[index] = pt;
                    grid_num[cam_idx][index] = TYPE_POINTCLOUD;
                }
            }
        }
    }

    int add = 0;
    for (int i = 0; i < length; i++) 
    {
        for(size_t cam_idx = 0; cam_idx < cams.size(); ++cam_idx){
            if (grid_num[cam_idx][i] == TYPE_POINTCLOUD) // && (map_value[cam_idx][i] >= 10)) //! debug
            {
                V3D pt = add_voxel_points_[i];
                V2D pc = new_frame_->w2c(pt, cam_idx); // 修改 w2c 以支持多相机

                PointPtr pt_new = std::make_shared<Point>(pt);
                Vector3d f = cams[cam_idx]->cam2world(pc);
                FeaturePtr ftr_new = std::make_shared<Feature>(pc, f, new_frame_->T_f_w_, map_value[cam_idx][i], 0);
                ftr_new->img = new_frame_->img_pyr_[0];
                ftr_new->id_ = new_frame_->id_;

                pt_new->addFrameRef(ftr_new);
                pt_new->value = map_value[cam_idx][i];
                AddPoint(pt_new);
                add += 1;
            }
        }
    }

    printf("[ VIO ]: Add %d 3D points.\n", add);
}

void LidarSelector::AddPoint(PointPtr pt_new)
{
    V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
    double voxel_size = 0.5;
    int loc_xyz[3];
    for(int j = 0; j < 3; j++)
    {
        loc_xyz[j] = floor(pt_w[j] / voxel_size);
        if(loc_xyz[j] < 0)
        {
            loc_xyz[j] -= 1;
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
    const int level_ref,    // the corresponding pyramid level of px_ref
    const int pyramid_level,
    const int halfpatch_size,
    Matrix2d& A_cur_ref)
{
    // Compute affine warp matrix A_ref_cur
    const Vector3d xyz_ref(f_ref * depth_ref);
    Vector3d xyz_du_ref = cam.cam2world(px_ref + Vector2d(halfpatch_size, 0) * (1 << level_ref) * (1 << pyramid_level));
    Vector3d xyz_dv_ref = cam.cam2world(px_ref + Vector2d(0, halfpatch_size) * (1 << level_ref) * (1 << pyramid_level));
    xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
    xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];
    const Vector2d px_cur = cam.world2cam(T_cur_ref * xyz_ref);
    const Vector2d px_du = cam.world2cam(T_cur_ref * xyz_du_ref);
    const Vector2d px_dv = cam.world2cam(T_cur_ref * xyz_dv_ref);
    A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
    A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
}

void LidarSelector::warpAffine(
    const Matrix2d& A_cur_ref,
    const cv::Mat& img_ref,
    const Vector2d& px_ref,
    const int level_ref,
    const int search_level,
    const int pyramid_level,
    const int halfpatch_size,
    float* patch)
{
    const int patch_size = halfpatch_size * 2;
    const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
    if(isnan(A_ref_cur(0,0)))
    {
        printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
        return;
    }
    
    for (int y = 0; y < patch_size; ++y)
    {
        for (int x = 0; x < patch_size; ++x)
        {
            Vector2f px_patch(x - halfpatch_size, y - halfpatch_size);
            px_patch *= (1 << search_level);
            px_patch *= (1 << pyramid_level);
            const Vector2f px = A_ref_cur * px_patch + px_ref.cast<float>();
            if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1)
                patch[patch_size_total * pyramid_level + y * patch_size + x] = 0;
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

    double numerator = 0.0, denominator1 = 0.0, denominator2 = 0.0;
    for (int i = 0; i < patch_size; i++) 
    {
        double diff_ref = ref_patch[i] - mean_ref;
        double diff_cur = cur_patch[i] - mean_curr;
        numerator += diff_ref * diff_cur;
        denominator1 += diff_ref * diff_ref;
        denominator2 += diff_cur * diff_cur;
    }
    return numerator / (sqrt(denominator1 * denominator2) + 1e-10);
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
    for(int y = 1; y < patch_size + 1; ++y, ref_patch_ptr += patch_size)
    {
        float* ref_patch_border_ptr = patch_with_border + y * (patch_size + 2) + 1;
        for(int x = 0; x < patch_size; ++x)
            ref_patch_ptr[x] = ref_patch_border_ptr[x];
    }
}
#endif

void LidarSelector::addFromSparseMap(cv::Mat img, PointCloudXYZI::Ptr pg)
{
    // 假设此方法用于从稀疏地图添加数据
    if(feat_map.empty()) return;

    // 下采样点云
    pg_down->reserve(feat_map.size());
    downSizeFilter.setInputCloud(pg);
    downSizeFilter.filter(*pg_down);
    
    reset_grid();
    size_t num_cameras = cams.size();
    for(size_t cam_idx = 0; cam_idx < num_cameras; ++cam_idx){
        std::fill_n(map_value[cam_idx].get(), length, 0.0f);
    }

    sub_sparse_map->reset();
    sub_map_cur_frame_.clear();

    float voxel_size = 0.5f;
    
    std::unordered_map<VOXEL_KEY, float> sub_feat_map;
    std::unordered_map<int, Warp*> Warp_map;

    // 创建深度图，每个相机一个
    std::vector<cv::Mat> depth_imgs(cams.size(), cv::Mat::zeros(height, width, CV_32FC1));
    std::vector<float*> depth_it(cams.size(), nullptr);
    for(size_t cam_idx = 0; cam_idx < cams.size(); ++cam_idx){
        depth_it[cam_idx] = (float*)depth_imgs[cam_idx].data;
    }

    for(int i = 0; i < pg_down->size(); i++)
    {
        V3D pt_w(pg_down->points[i].x, pg_down->points[i].y, pg_down->points[i].z);

        // 确定哈希表的键
        int loc_xyz[3];
        for(int j = 0; j < 3; j++)
        {
            loc_xyz[j] = floor(pt_w[j] / voxel_size);
        }
        VOXEL_KEY position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

        sub_feat_map[position] = 1.0f; // 只记录存在

        // 处理所有相机
        for(size_t cam_idx = 0; cam_idx < cams.size(); ++cam_idx){
            V3D pt_cam = new_frame_->w2f(pt_w, cam_idx); // 修改 w2f 以支持多相机

            if(pt_cam[2] > 0)
            {
                V2D pc = cams[cam_idx]->world2cam(Tcw[cam_idx] * pt_w + Pcw[cam_idx]);
                float u = fx[cam_idx] * pt_cam[0] / pt_cam[2] + cx[cam_idx];
                float v = fy[cam_idx] * pt_cam[1] / pt_cam[2] + cy[cam_idx];

                if(cams[cam_idx]->isInFrame(pc.cast<int>(), (patch_size_half + 1) * 8))
                {
                    int col = static_cast<int>(pc[0]);
                    int row = static_cast<int>(pc[1]);
                    if(col < 0 || col >= width || row < 0 || row >= height){
                        continue;
                    }
                    depth_it[cam_idx][width * row + col] = pt_cam[2];
                }
            }
        }
    }

    // 关联特征点
    for(auto& iter : sub_feat_map)
    {   
        VOXEL_KEY position = iter.first;
        auto corre_voxel = feat_map.find(position);

        if(corre_voxel != feat_map.end())
        {
            std::vector<PointPtr> &voxel_points = corre_voxel->second->voxel_points;
            int voxel_num = voxel_points.size();
            for (int i = 0; i < voxel_num; i++)
            {
                PointPtr pt = voxel_points[i];
                if(pt == nullptr) continue;
                // 假设第一个相机
                V3D pt_cam = new_frame_->w2f(pt->pos_, 0); // 修改 w2f 以支持多相机，假设第一个相机
                
                if(pt_cam[2] < 0) continue;

                V2D pc = new_frame_->w2c(pt->pos_, 0); // 修改 w2c 以支持多相机

                FeaturePtr ref_ftr;

                if(cams[0]->isInFrame(pc.cast<int>(), (patch_size_half + 1) * 8))
                {
                    int index = static_cast<int>(pc[0] / grid_size) * grid_n_height + static_cast<int>(pc[1] / grid_size);
                    grid_num[0][index] = TYPE_MAP;
                    Vector3d obs_vec(new_frame_->pos() - pt->pos_);

                    float cur_dist = obs_vec.norm();
                    float cur_value = pt->value;

                    if (cur_dist <= map_dist[0][index]) 
                    {
                        map_dist[0][index] = cur_dist;
                        voxel_points_[index] = pt;
                    } 

                    if (cur_value >= map_value[0][index])
                    {
                        map_value[0][index] = cur_value;
                    }
                }
            }    
        } 
    }
        
    // 进一步处理
    for (int i = 0; i < length; i++) 
    { 
        for(size_t cam_idx = 0; cam_idx < cams.size(); ++cam_idx){
            if (grid_num[cam_idx][i] == TYPE_MAP) // && map_value[cam_idx][i] > 10)
            {
                PointPtr pt = voxel_points_[i];
                if(pt == nullptr) continue;

                V2D pc = new_frame_->w2c(pt->pos_, cam_idx); // 修改 w2c 以支持多相机
                V3D pt_cam = new_frame_->w2f(pt->pos_, cam_idx); // 修改 w2f 以支持多相机

                bool depth_continous = false;
                for (int u = -patch_size_half; u <= patch_size_half; u++)
                {
                    for (int v = -patch_size_half; v <= patch_size_half; v++)
                    {
                        if(u == 0 && v == 0) continue;

                        float depth = depth_imgs[cam_idx].at<float>(v + static_cast<int>(pc[1]), u + static_cast<int>(pc[0]));

                        if(depth == 0.0f) continue;

                        double delta_dist = abs(pt_cam[2] - depth);

                        if(delta_dist > 1.5)
                        {                
                            depth_continous = true;
                            break;
                        }
                    }
                    if(depth_continous) break;
                }
                if(depth_continous) continue;

                FeaturePtr ref_ftr;

                if(!pt->getCloseViewObs(new_frame_->pos(), ref_ftr, pc, cam_idx)) continue; // 修改 getCloseViewObs 以支持多相机

                std::vector<float> patch_wrap(patch_size_total * 3);

                // 获取仿射变换矩阵
                int search_level = 0;
                Matrix2d A_cur_ref_zero;

                auto iter_warp = Warp_map.find(ref_ftr->id_);
                if(iter_warp != Warp_map.end())
                {
                    search_level = iter_warp->second->search_level;
                    A_cur_ref_zero = iter_warp->second->A_cur_ref;
                }
                else
                {
                    getWarpMatrixAffine(*cams[cam_idx], ref_ftr->px, ref_ftr->f, (ref_ftr->pos() - pt->pos_).norm(), 
                    new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(), 0, 0, patch_size_half, A_cur_ref_zero);
                    
                    search_level = getBestSearchLevel(A_cur_ref_zero, 2);

                    Warp *ot = new Warp(search_level, A_cur_ref_zero);
                    Warp_map[ref_ftr->id_] = ot;
                }

                // 仿射变换补丁
                for(int pyramid_level = 0; pyramid_level <= 2; pyramid_level++)
                {                
                    warpAffine(A_cur_ref_zero, ref_ftr->img, ref_ftr->px, ref_ftr->level, search_level, pyramid_level, patch_size_half, patch_wrap.data());
                }

                getpatch(img, pc, patch_cache[cam_idx].data(), 0, cam_idx);

                if(ncc_en)
                {
                    double ncc = NCC(patch_wrap.data(), patch_cache[cam_idx].data(), patch_size_total);
                    if(ncc < ncc_thre) continue;
                }

                float error = 0.0f;
                for (int ind = 0; ind < patch_size_total; ind++) 
                {
                    error += (patch_wrap[ind] - patch_cache[cam_idx][ind]) * (patch_wrap[ind] - patch_cache[cam_idx][ind]);
                }
                if(error > outlier_threshold * patch_size_total) continue;
                
                sub_map_cur_frame_.push_back(pt);

                sub_sparse_map->propa_errors.push_back(error);
                sub_sparse_map->search_levels.push_back(search_level);
                sub_sparse_map->errors.push_back(error);
                sub_sparse_map->index.push_back(i);  
                sub_sparse_map->voxel_points.push_back(pt);
                sub_sparse_map->patch.push_back(std::move(patch_wrap));
            }
        }
    }

void LidarSelector::AddObservation(cv::Mat img)
{
    size_t num_cameras = cams.size();
    int total_points = sub_sparse_map->index.size();
    if (total_points == 0) return;

    for(size_t cam_idx = 0; cam_idx < num_cameras; ++cam_idx){
        memset(align_flag[cam_idx].get(), 0, sizeof(int) * length);
        int FeatureAlignmentNum = 0;

        for (int i = 0; i < total_points; i++) 
        {
            bool res;
            int search_level = sub_sparse_map->search_levels[i];
            Vector2d px_scaled = sub_sparse_map->px_cur[i] / (1 << search_level);
            res = align2D(img, sub_sparse_map->patch_with_border[i], sub_sparse_map->patch[i],
                            20, px_scaled, i);
            sub_sparse_map->px_cur[i] = px_scaled * (1 << search_level);
            if(res)
            {
                align_flag[cam_idx][i] = 1;
                FeatureAlignmentNum++;
            }
        }
    }
}

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
    bool converged = false;

    // compute derivative of template and prepare inverse compositional
    float ref_patch_dx[patch_area_] __attribute__((aligned(16)));
    float ref_patch_dy[patch_area_] __attribute__((aligned(16)));
    Matrix3f H; H.setZero();

    // compute gradient and hessian
    const int ref_step = patch_size_ + 2;
    float* it_dx = ref_patch_dx;
    float* it_dy = ref_patch_dy;
    for(int y = 0; y < patch_size_; y++) 
    {
        float* it = ref_patch_with_border + (y + 1) * ref_step + 1; 
        for(int x = 0; x < patch_size_; x++, ++it_dx, ++it_dy)
        {
            Vector3f J;
            J[0] = 0.5f * (it[1] - it[-1]); 
            J[1] = 0.5f * (it[ref_step] - it[-ref_step]); 
            J[2] = 1.0f; 
            *it_dx = J[0];
            *it_dy = J[1];
            H += J * J.transpose(); 
        }
    }
    Matrix3f Hinv = H.inverse();
    float mean_diff = 0.0f;

    // Compute pixel location in new image:
    float u = cur_px_estimate.x();
    float v = cur_px_estimate.y();

    // termination condition
    const float min_update_squared = 0.03f * 0.03f;
    const int cur_step = cur_img.step.p[0];
    float chi2 = sub_sparse_map->propa_errors[index];
    Vector3f update; update.setZero();
    for(int iter = 0; iter < n_iter; iter++)
    {
        int u_r = floorf(u);
        int v_r = floorf(v);
        if(u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols - halfpatch_size_ || v_r >= cur_img.rows - halfpatch_size_)
            break;

        if(isnan(u) || isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
            return false;

        // compute interpolation weights
        float subpix_x = u - u_r;
        float subpix_y = v - v_r;
        float wTL = (1.0f - subpix_x) * (1.0f - subpix_y);
        float wTR = subpix_x * (1.0f - subpix_y);
        float wBL = (1.0f - subpix_x) * subpix_y;
        float wBR = subpix_x * subpix_y;

        // loop through search_patch, interpolate
        float* it_ref = ref_patch;
        float* it_ref_dx = ref_patch_dx;
        float* it_ref_dy = ref_patch_dy;
        float new_chi2 = 0.0f;
        Vector3f Jres; Jres.setZero();
        for(int y = 0; y < patch_size_; y++)
        {
            uint8_t* it = (uint8_t*) cur_img.data + (v_r + y - halfpatch_size_) * cur_step + (u_r - halfpatch_size_); 
            for(int x = 0; x < patch_size_; x++, it++, ++it_ref, ++it_ref_dx, ++it_ref_dy)
            {
                float search_pixel = wTL * it[0] + wTR * it[1] + wBL * it[cur_step] + wBR * it[cur_step + 1];
                float res = search_pixel - *it_ref + mean_diff;
                Jres[0] -= res * (*it_ref_dx);
                Jres[1] -= res * (*it_ref_dy);
                Jres[2] -= res;
                new_chi2 += res * res;
            }
        }

        if(iter > 0 && new_chi2 > chi2)
        {
            // cout << "error increased." << endl;
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

        if(update[0]*update[0] + update[1]*update[1] < min_update_squared)
        {
            converged = true;
            break;
        }
    }

    cur_px_estimate << u, v;
    return converged;
}

void LidarSelector::ComputeJ(cv::Mat img) 
{
    int total_points = sub_sparse_map->index.size();
    if (total_points == 0) return;
    float error = 1e10f;
    float now_error = error;

    for (int level = 2; level >= 0; level--) 
    {
        now_error = UpdateState(img, error, level);
    }
    if (now_error < error)
    {
        state->cov -= G * state->cov;
    }
    updateFrameState(*state);
}

float LidarSelector::UpdateState(cv::Mat img, float total_residual, int level) 
{
    int total_points = sub_sparse_map->index.size();
    if (total_points == 0) return 0.0f;
    StatesGroup old_state = (*state);
    V2D pc; 
    MD(1,2) Jimg;
    MD(2,3) Jdpi;
    MD(1,3) Jdphi, Jdp, JdR, Jdt;
    VectorXd z;
    bool EKF_end = false;

    /* Compute J */
    float error = 0.0f, last_error = total_residual, patch_error = 0.0f, propa_error = 0.0f;
    const int H_DIM = total_points * patch_size_total;
    
    z.resize(H_DIM);
    z.setZero();

    H_sub.resize(H_DIM, 6);
    H_sub.setZero();
    
    for (int iteration = 0; iteration < NUM_MAX_ITERATIONS; iteration++) 
    {
        error = 0.0f;
        propa_error = 0.0f;
        n_meas_ = 0;
        M3D Rwi(state->rot_end);
        V3D Pwi(state->pos_end);
        Rcw = Rci * Rwi.transpose();
        Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
        Jdp_dt = Rci * Rwi.transpose();
        
        M3D p_hat;
        int i;

        for (i = 0; i < sub_sparse_map->index.size(); i++) 
        {
            patch_error = 0.0f;
            int search_level = sub_sparse_map->search_levels[i];
            int pyramid_level = level + search_level;
            const int scale =  (1 << pyramid_level);
            
            PointPtr pt = sub_sparse_map->voxel_points[i];

            if(pt == nullptr) continue;

            // 遍历所有相机
            for(size_t cam_idx = 0; cam_idx < cams.size(); ++cam_idx){
                V3D pf = Rcw * pt->pos_ + Pcw;
                V2D pc = cams[cam_idx]->world2cam(pf);
                
                dpi(pf, Jdpi, cam_idx); // 修改 dpi 方法以支持多相机
                p_hat << SKEW_SYM_MATRX(pf);
                
                const float u_ref = pc[0];
                const float v_ref = pc[1];
                const int u_ref_i = floorf(pc[0] / scale) * scale; 
                const int v_ref_i = floorf(pc[1] / scale) * scale;
                const float subpix_u_ref = (u_ref - u_ref_i) / scale;
                const float subpix_v_ref = (v_ref - v_ref_i) / scale;
                const float w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
                const float w_ref_tr = subpix_u_ref * (1.0f - subpix_v_ref);
                const float w_ref_bl = (1.0f - subpix_u_ref) * subpix_v_ref;
                const float w_ref_br = subpix_u_ref * subpix_v_ref;
                
                std::vector<float> P = sub_sparse_map->patch[i];
                for (int x = 0; x < patch_size; x++) 
                {
                    int img_x = u_ref_i - patch_size_half * scale + x * scale;
                    int img_y = v_ref_i - patch_size_half * scale;
                    
                    // 确保索引不越界
                    if(img_x < 0 || img_x >= width - scale || img_y < 0 || img_y >= height - scale){
                        continue;
                    }

                    uint8_t* img_ptr = img.data + (img_y * width + img_x);
                    for (int y = 0; y < patch_size; y++, img_ptr += scale)
                    {
                        float du = 0.5f * ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] + w_ref_br * img_ptr[scale * width + scale * 2])
                                    - (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
                        float dv = 0.5f * ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] + w_ref_br * img_ptr[width * scale * 2 + scale])
                                    - (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));
                        Jimg << du, dv;
                        Jimg = Jimg * (1.0f / scale);
                        Jdphi = Jimg * Jdpi * p_hat;
                        Jdp = -Jimg * Jdpi;
                        JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
                        Jdt = Jdp * Jdp_dt;
                        
                        double res = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale] - P[patch_size_total * level + x * patch_size + y];
                        z(i * patch_size_total + x * patch_size + y) = res;
                        patch_error += res * res;
                        n_meas_++;
                        H_sub.block<1,6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
                    }
                }  

                sub_sparse_map->errors[i] = patch_error;
                error += patch_error;
            }

            error = error / n_meas_;

            if (error <= last_error) 
            {
                old_state = (*state);
                last_error = error;

                auto &&H_sub_T = H_sub.transpose();
                H_T_H.block<6,6>(0,0) = H_sub_T * H_sub;
                MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
                auto &&HTz = H_sub_T * z;

                auto vec = (*state_propagat) - (*state);
                G.block<6,6>(0,0) = K_1.block<6,6>(0,0) * H_T_H.block<6,6>(0,0);
                auto solution = - K_1.block<6,6>(0,0) * HTz + vec - G.block<6,6>(0,0) * vec.block<6,1>(0,0);
                (*state) += solution;
                auto &&rot_add = solution.block<3,1>(0,0);
                auto &&t_add   = solution.block<3,1>(3,0);

                if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))
                {
                    EKF_end = true;
                }
            }
            else
            {
                (*state) = old_state;
                EKF_end = true;
            }

            if (iteration == NUM_MAX_ITERATIONS || EKF_end) 
            {
                break;
            }
        }
    return last_error;
} 

void LidarSelector::updateFrameState(StatesGroup state)
{
    M3D Rwi(state.rot_end);
    V3D Pwi(state.pos_end);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
    new_frame_->T_f_w_ = SE3(Rcw, Pcw);
}

void LidarSelector::addObservation(cv::Mat img)
{
    size_t num_cameras = cams.size();
    int total_points = sub_sparse_map->index.size();
    if (total_points == 0) return;

    for(size_t cam_idx = 0; cam_idx < num_cameras; ++cam_idx){
        memset(align_flag[cam_idx].get(), 0, sizeof(int) * length);
        int FeatureAlignmentNum = 0;

        for (int i = 0; i < total_points; i++) 
        {
            bool res;
            int search_level = sub_sparse_map->search_levels[i];
            Vector2d px_scaled = sub_sparse_map->px_cur[i] / (1 << search_level);
            res = align2D(img, sub_sparse_map->patch_with_border[i], sub_sparse_map->patch[i],
                            20, px_scaled, i);
            sub_sparse_map->px_cur[i] = px_scaled * (1 << search_level);
            if(res)
            {
                align_flag[cam_idx][i] = 1;
                FeatureAlignmentNum++;
            }
        }
    }
}

V3F LidarSelector::getpixel(cv::Mat img, V2D pc, int cam_idx, int width, int height) 
{
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(u_ref); 
    const int v_ref_i = floorf(v_ref);
    const float subpix_u_ref = u_ref - u_ref_i;
    const float subpix_v_ref = v_ref - v_ref_i;
    const float w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0f - subpix_v_ref);
    const float w_ref_bl = (1.0f - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    
    // 确保索引不越界
    if(u_ref_i <= 0 || v_ref_i <= 0 || u_ref_i >= width -1 || v_ref_i >= height -1){
        return V3F(0.0f, 0.0f, 0.0f);
    }

    uint8_t* img_ptr = img.data + ((v_ref_i) * width + (u_ref_i)) * 3;
    float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[3] + w_ref_bl * img_ptr[width * 3] + w_ref_br * img_ptr[width * 3 + 3];
    float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[4] + w_ref_bl * img_ptr[width * 3 + 1] + w_ref_br * img_ptr[width * 3 + 4];
    float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[5] + w_ref_bl * img_ptr[width * 3 + 2] + w_ref_br * img_ptr[width * 3 + 5];
    V3F pixel(B, G, R);
    return pixel;
}



void LidarSelector::detect(cv::Mat img, PointCloudXYZI::Ptr pg) 
{
    if(width != img.cols || height != img.rows)
    {
        // 调整图像大小以匹配预期尺寸
        double scale = 0.5;
        cv::resize(img, img, cv::Size(static_cast<int>(img.cols * scale), static_cast<int>(img.rows * scale)), 0, 0, cv::INTER_LINEAR);
    }
    img_rgb = img.clone();
    img_cp = img.clone();
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    new_frame_.reset(new Frame(cams, img.clone()));
    updateFrameState(*state);

    if(stage_ == STAGE_FIRST_FRAME && pg->size() > 10)
    {
        new_frame_->setKeyframe();
        stage_ = STAGE_DEFAULT_FRAME;
    }

    double t1 = omp_get_wtime();

    addFromSparseMap(img, pg);

    double t3 = omp_get_wtime();

    addSparseMap(img, pg);

    double t4 = omp_get_wtime();
    
    ComputeJ(img);

    double t5 = omp_get_wtime();

    addObservation(img);
    
    double t2 = omp_get_wtime();
    
    frame_count++;
    ave_total = ave_total * (frame_count - 1) / frame_count + (t2 - t1) / frame_count;

    printf("[ VIO ]: time: addFromSparseMap: %.6f addSparseMap: %.6f ComputeJ: %.6f addObservation: %.6f total time: %.6f ave_total: %.6f.\n"
    , t3 - t1, t4 - t3, t5 - t4, t2 - t5, t2 - t1, ave_total);

    display_keypatch(t2 - t1);
} 

} // namespace lidar_selection
