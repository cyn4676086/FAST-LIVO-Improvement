#include "lidar_selection.h"

namespace lidar_selection {

LidarSelector::LidarSelector(const int gridsize, SparseMap* sparsemap )
    : grid_size(gridsize), sparse_map(sparsemap), ave_total(0.0), frame_count(0)
{
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    G = Matrix<double, DIM_STATE, DIM_STATE>::Zero();
    H_T_H = Matrix<double, DIM_STATE, DIM_STATE>::Zero();
    Rli = M3D::Identity();
    Rci = M3D::Identity();
    Rcw = M3D::Identity();
    Jdphi_dR = M3D::Identity();
    Jdp_dt = M3D::Identity();
    Jdp_dR = M3D::Identity();
    Pli = V3D::Zero();
    Pci = V3D::Zero();
    Pcw = V3D::Zero();
    stage_ = STAGE_FIRST_FRAME;
    weight_scale_ = 10;
    scale_estimator_.reset(new vk::robust_cost::UnitScaleEstimator());
    weight_function_.reset(new vk::robust_cost::HuberWeightFunction());
    NUM_MAX_ITERATIONS = 10;
    ncc_en = false; 
    outlier_threshold = 1000.0;
    ncc_thre = 0.5; 
    img_point_cov = 1.0;
    patch_size = 8; 
    patch_size_total = patch_size*patch_size;
    patch_size_half = patch_size/2;
    debug = 0;
    MIN_IMG_COUNT = 1;
}

LidarSelector::~LidarSelector() 
{
    delete sparse_map;
    delete sub_sparse_map;
    delete[] grid_num;
    delete[] map_index;
    delete[] map_value;
    free(map_dist);
    for (auto &kv : feat_map)
    {
        delete kv.second;
    }
    feat_map.clear();
    Warp_map.clear();
    sub_feat_map.clear();
}

void LidarSelector::set_extrinsic(const V3D &transl, const M3D &rot)
{
    Pli = -rot.transpose() * transl;
    Rli = rot.transpose();
}

void LidarSelector::init()
{
    sub_sparse_map = new SubSparseMap;

    Rci = sparse_map->Rcl * Rli;
    Pci= sparse_map->Rcl*Pli + sparse_map->Pcl;

    M3D Pic_skew;
    V3D Pic = -Rci.transpose() * Pci;
    Pic_skew << SKEW_SYM_MATRX(Pic);
    Jdp_dR = -Rci * Pic_skew;

    // 多相机修改：对每个相机提取宽高和参数
    widths_.resize(cams_.size());
    heights_.resize(cams_.size());
    fxs_.resize(cams_.size());
    fys_.resize(cams_.size());
    cxs_.resize(cams_.size());
    cys_.resize(cams_.size());

    for (size_t i = 0; i < cams_.size(); i++) {
        widths_[i] = cams_[i]->width();
        heights_[i] = cams_[i]->height();
        // 假设抽取fx, fy, cx, cy的方法与单相机相同，根据实际相机模型修改
        // 这里以errorMultiplier2()与errorMultiplier()作为参考，需要根据camera model实际方法修改
        fxs_[i] = cams_[i]->errorMultiplier2();
        fys_[i] = cams_[i]->errorMultiplier()/ (4.0 * fxs_[i]);
        cxs_[i] = (widths_[i]-1)/2.0;
        cys_[i] = (heights_[i]-1)/2.0; 
    }

    // 假设只使用同一grid参数，但如果需要为每个相机单独维护grid可自行扩展
    width = widths_[0];
    height = heights_[0];
    grid_n_width = static_cast<int>(width/grid_size);
    grid_n_height = static_cast<int>(height/grid_size);
    length = grid_n_width * grid_n_height;

    grid_num = new int[length];
    map_index = new int[length];
    map_value = new float[length];
    map_dist = (float*)malloc(sizeof(float)*length);
    memset(grid_num, TYPE_UNKNOWN, sizeof(int)*length);
    memset(map_index, 0, sizeof(int)*length);
    memset(map_value, 0, sizeof(float)*length);
    voxel_points_.reserve(length);
    add_voxel_points_.reserve(length);
}

void LidarSelector::reset_grid()
{
    memset(grid_num, TYPE_UNKNOWN, sizeof(int)*length);
    memset(map_index, 0, sizeof(int)*length);
    std::fill_n(map_dist, length, 10000.0f);
    std::vector<PointPtr>(length).swap(voxel_points_);
    std::vector<V3D>(length).swap(add_voxel_points_);
    voxel_points_.reserve(length);
    add_voxel_points_.reserve(length);
}

// 多相机修改：CheckGoodPoints现在需要使用对应相机的内参与图像
float LidarSelector::CheckGoodPoints(cv::Mat img, V2D uv, int cam_id)
{
    const float u_ref = uv[0];
    const float v_ref = uv[1];
    const int u_ref_i = floorf(uv[0]); 
    const int v_ref_i = floorf(uv[1]);
    if(u_ref_i < 1 || u_ref_i >= widths_[cam_id]-1 || v_ref_i < 1 || v_ref_i >= heights_[cam_id]-1)
        return 0.0f;
    const float subpix_u_ref = u_ref-u_ref_i;
    const float subpix_v_ref = v_ref-v_ref_i;
    uint8_t* img_ptr = (uint8_t*) img.data + (v_ref_i)*widths_[cam_id] + (u_ref_i);
    float gu = 2*(img_ptr[1] - img_ptr[-1]) + img_ptr[1-widths_[cam_id]] - img_ptr[-1-widths_[cam_id]] + img_ptr[1+widths_[cam_id]] - img_ptr[-1+widths_[cam_id]];
    float gv = 2*(img_ptr[widths_[cam_id]] - img_ptr[-widths_[cam_id]]) + img_ptr[widths_[cam_id]+1] - img_ptr[-widths_[cam_id]+1] + img_ptr[widths_[cam_id]-1] - img_ptr[-widths_[cam_id]-1];
    return fabs(gu)+fabs(gv);
}

// 多相机修改：getpatch以相机参数为基础
void LidarSelector::getpatch(cv::Mat img, V2D pc, float* patch_tmp, int level, int cam_id) 
{
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int scale =  (1<<level);
    const int u_ref_i = floorf(u_ref/scale)*scale; 
    const int v_ref_i = floorf(v_ref/scale)*scale;
    const float subpix_u_ref = (u_ref-u_ref_i)/scale;
    const float subpix_v_ref = (v_ref-v_ref_i)/scale;
    const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
    const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;

    for (int x=0; x<patch_size; x++) 
    {
        int row = v_ref_i - patch_size_half*scale + x*scale;
        if(row < 0 || row >= heights_[cam_id]-scale) continue;
        uint8_t* img_ptr = (uint8_t*) img.data + row*widths_[cam_id] + (u_ref_i-patch_size_half*scale);
        for (int y=0; y<patch_size; y++, img_ptr+=scale)
        {
            int col = u_ref_i - patch_size_half*scale + y*scale;
            if(col < 0 || col >= widths_[cam_id]-scale) 
            {
                patch_tmp[patch_size_total*level+x*patch_size+y] = 0;
                continue;
            }
            patch_tmp[patch_size_total*level+x*patch_size+y] = 
                w_ref_tl*img_ptr[0] + 
                w_ref_tr*img_ptr[scale] + 
                w_ref_bl*img_ptr[scale*widths_[cam_id]] + 
                w_ref_br*img_ptr[scale*widths_[cam_id]+scale];
        }
    }
}

// dpi函数无需相机区分
void LidarSelector::dpi(V3D p, MD(2,3)& J) {
    const double x = p[0];
    const double y = p[1];
    const double z_inv = 1./p[2];
    const double z_inv_2 = z_inv * z_inv;
    // 默认使用当前处理的相机fx, fy时在ComputeJ中根据cam_id换
    // 这里J只是对xyz到uv的基本投影Jacobian，不乘fx,fy
    J(0,0) = z_inv;
    J(0,1) = 0.0;
    J(0,2) = - x * z_inv_2;
    J(1,0) = 0.0;
    J(1,1) = z_inv;
    J(1,2) = - y * z_inv_2;
}

// 多相机修改：NCC无需相机参数，但请注意patch_size_total不变
double LidarSelector::NCC(float* ref_patch, float* cur_patch, int patch_size)
{    
    double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
    double mean_ref =  sum_ref / patch_size;

    double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
    double mean_curr =  sum_cur / patch_size;

    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < patch_size; i++) 
    {
        double nr = (ref_patch[i] - mean_ref);
        double nc = (cur_patch[i] - mean_curr);
        double n = nr*nc;
        numerator += n;
        demoniator1 += nr*nr;
        demoniator2 += nc*nc;
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

// getWarpMatrixAffine、warpAffine、getBestSearchLevel等函数与相机参数相关时需传cam_id，以获取fx, fy
void LidarSelector::getWarpMatrixAffine(
    const vk::AbstractCamera& cam,
    const Vector2d& px_ref,
    const Vector3d& f_ref,
    const double depth_ref,
    const Sophus::SE3d& T_cur_ref,
    const int level_ref,
    const int pyramid_level,
    const int halfpatch_size,
    Matrix2d& A_cur_ref)
{
  Vector3d xyz_ref = f_ref*depth_ref;
  Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size,0)*(1<<level_ref)*(1<<pyramid_level)));
  Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0,halfpatch_size)*(1<<level_ref)*(1<<pyramid_level)));
  xyz_du_ref *= xyz_ref[2]/xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2]/xyz_dv_ref[2];
  const Vector2d px_cur(cam.world2cam(T_cur_ref*(xyz_ref)));
  const Vector2d px_du(cam.world2cam(T_cur_ref*(xyz_du_ref)));
  const Vector2d px_dv(cam.world2cam(T_cur_ref*(xyz_dv_ref)));
  A_cur_ref.col(0) = (px_du - px_cur)/halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur)/halfpatch_size;
}

void LidarSelector::warpAffine(
    const Matrix2d& A_cur_ref,
    const cv::Mat& img_ref,
    const Vector2d& px_ref,
    const int level_ref,
    const int search_level,
    const int pyramid_level,
    const int halfpatch_size,
    float* patch,
    int cam_id)
{
  const int patch_size = halfpatch_size*2 ;
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if(std::isnan(A_ref_cur(0,0)))
  {
    return;
  }
  for (int y=0; y<patch_size; ++y)
  {
    for (int x=0; x<patch_size; ++x)
    {
      Vector2f px_patch(x-halfpatch_size, y-halfpatch_size);
      px_patch *= (1<<search_level)*(1<<pyramid_level);
      const Vector2f px(A_ref_cur*px_patch + px_ref.cast<float>());
      if (px[0]<0 || px[1]<0 || px[0]>=img_ref.cols-1 || px[1]>=img_ref.rows-1)
        patch[patch_size_total*pyramid_level + y*patch_size+x] = 0;
      else
        patch[patch_size_total*pyramid_level + y*patch_size+x] = (float) vk::interpolateMat_8u(img_ref, px[0], px[1]);
    }
  }
}

int LidarSelector::getBestSearchLevel(
    const Matrix2d& A_cur_ref,
    const int max_level)
{
  int search_level = 0;
  double D = A_cur_ref.determinant();
  while(D > 3.0 && search_level < max_level)
  {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

// 多相机扩展：AddPoint与地图无关，原逻辑不变
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

// 以下两个函数 addFromSparseMap 和 addSparseMap 原逻辑是对单相机图像 img 进行处理
// 多相机扩展：对于每个相机的图像，我们都要执行类似的逻辑，将结果累积到sub_sparse_map和voxel_points_中
// 同理，Patch提取、NCC等操作都需要使用相应的camera参数(cams_[cam_id])和imgs[cam_id]

void LidarSelector::addSparseMap(cv::Mat img, pcl::PointCloud<PointType>::Ptr pg, int cam_id) 
{
    reset_grid();
    for (int i=0; i<pg->size(); i++) 
    {
        V3D pt(pg->points[i].x, pg->points[i].y, pg->points[i].z);
        V2D pc(new_frame_->w2c(pt, cam_id));
        if(cams_[cam_id]->isInFrame(pc.cast<int>(), (patch_size_half+1)*8)) 
        {
            int index = static_cast<int>(pc[0]/grid_size)*grid_n_height + static_cast<int>(pc[1]/grid_size);
            float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]); // 这里仍然用shiTomasiScore
            if (cur_value > map_value[index])
            {
                map_value[index] = cur_value;
                add_voxel_points_[index] = pt;
                grid_num[index] = TYPE_POINTCLOUD;
            }
        }
    }
    int add=0;
    for (int i=0; i<length; i++) 
    {
        if (grid_num[i]==TYPE_POINTCLOUD)
        {
            V3D pt = add_voxel_points_[i];
            V2D pc(new_frame_->w2c(pt, cam_id));
            PointPtr pt_new(new Point(pt));
            Vector3d f = cams_[cam_id]->cam2world(pc[0], pc[1]);
            FeaturePtr ftr_new(new Feature(pc, f, new_frame_->T_f_w_, map_value[i], 0, cam_id));
            ftr_new->img = new_frame_->img_pyr_[cam_id][0];
            ftr_new->id_ = new_frame_->id_;
            pt_new->addFrameRef(ftr_new);
            pt_new->value = map_value[i];
            AddPoint(pt_new);
            add += 1;
        }
    }
    printf("[ VIO ]: Add %d 3D points from camera %d.\n", add, cam_id);
}


void LidarSelector::addFromSparseMap(cv::Mat img, pcl::PointCloud<PointType>::Ptr pg, int cam_id)
{
    if(feat_map.size()<=0) return;

    pg_down.reset(new pcl::PointCloud<PointType>());
    downSizeFilter.setInputCloud(pg);
    downSizeFilter.filter(*pg_down);

    reset_grid();
    memset(map_value, 0, sizeof(float)*length);

    sub_sparse_map->reset();
    std::deque< PointPtr >().swap(sub_map_cur_frame_);

    float voxel_size = 0.5;

    sub_feat_map.clear();
    Warp_map.clear();

    cv::Mat depth_img = cv::Mat::zeros(heights_[cam_id], widths_[cam_id], CV_32FC1);
    float* it = (float*)depth_img.data;

    // 构建深度图
    for(int i=0; i<pg_down->size(); i++)
    {
        V3D pt_w(pg_down->points[i].x, pg_down->points[i].y, pg_down->points[i].z);
        for(int j=0; j<3; j++)
        {
            // 计算voxel位置
        }
        VOXEL_KEY position((int64_t)floor(pt_w[0]/voxel_size), (int64_t)floor(pt_w[1]/voxel_size), (int64_t)floor(pt_w[2]/voxel_size));
        sub_feat_map[position] = 1.0f;

        V3D pt_c(new_frame_->w2f(pt_w, cam_id));
        if(pt_c[2] > 0)
        {
            double fx = fxs_[cam_id], fy = fys_[cam_id], cx = cxs_[cam_id], cy = cys_[cam_id];
            float u = fx * pt_c[0]/pt_c[2] + cx;
            float v = fy * pt_c[1]/pt_c[2] + cy;
            int col = (int)u;
            int row = (int)v;
            if(row>=0 && row<heights_[cam_id] && col>=0 && col<widths_[cam_id])
                it[widths_[cam_id]*row+col] = pt_c[2];        
        }
    }

    // 遍历sub_feat_map
    for(auto& iter : sub_feat_map)
    {   
        VOXEL_KEY position = iter.first;
        auto corre_voxel = feat_map.find(position);
        if(corre_voxel != feat_map.end())
        {
            std::vector<PointPtr> &voxel_points = corre_voxel->second->voxel_points;
            int voxel_num = voxel_points.size();
            for (int i=0; i<voxel_num; i++)
            {
                PointPtr pt = voxel_points[i];
                if(pt==nullptr) continue;
                V3D pt_cam(new_frame_->w2f(pt->pos_, cam_id));
                if(pt_cam[2]<0) continue;

                V2D pc(new_frame_->w2c(pt->pos_, cam_id));
                if(cams_[cam_id]->isInFrame(pc.cast<int>(), (patch_size_half+1)*8))
                {
                    int index = static_cast<int>(pc[0]/grid_size)*grid_n_height + static_cast<int>(pc[1]/grid_size);
                    grid_num[index] = TYPE_MAP;
                    Vector3d obs_vec(new_frame_->pos() - pt->pos_);
                    float cur_dist = obs_vec.norm();
                    float cur_value = pt->value;
                    if (cur_dist <= map_dist[index]) 
                    {
                        map_dist[index] = cur_dist;
                        voxel_points_[index] = pt;
                    } 
                    if (cur_value >= map_value[index])
                    {
                        map_value[index] = cur_value;
                    }
                }
            }    
        } 
    }

    for (int i=0; i<length; i++) 
    { 
        if (grid_num[i]==TYPE_MAP)
        {
            PointPtr pt = voxel_points_[i];
            if(pt==nullptr) continue;
            V2D pc(new_frame_->w2c(pt->pos_, cam_id));
            V3D pt_cam(new_frame_->w2f(pt->pos_, cam_id));

            // 检查深度连续性
            bool depth_continous = false;
            for (int u=-patch_size_half; u<=patch_size_half; u++)
            {
                for (int v=-patch_size_half; v<=patch_size_half; v++)
                {
                    if(u==0 && v==0) continue;
                    int r = v+(int)pc[1], c = u+(int)pc[0];
                    if(r<0 || r>=heights_[cam_id] || c<0 || c>=widths_[cam_id]) continue;
                    float depth = it[widths_[cam_id]*r+c];
                    if(depth == 0.) continue;
                    double delta_dist = fabs(pt_cam[2]-depth);
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
            if(!pt->getCloseViewObs(new_frame_->pos(), ref_ftr, pc)) continue;

            std::vector<float> patch_wrap(patch_size_total * 3);
            // 计算仿射warp
            Matrix2d A_cur_ref_zero;
            int search_level;
            auto iter_warp = Warp_map.find(ref_ftr->id_);
            if(iter_warp != Warp_map.end())
            {
                search_level = iter_warp->second->search_level;
                A_cur_ref_zero = iter_warp->second->A_cur_ref;
            }
            else
            {
                getWarpMatrixAffine(*cams_[ref_ftr->camera_id], ref_ftr->px, ref_ftr->f, (ref_ftr->pos() - pt->pos_).norm(), 
                new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(), 0, 0, patch_size_half, A_cur_ref_zero);
                search_level = getBestSearchLevel(A_cur_ref_zero, 2);
                Warp *ot = new Warp(search_level, A_cur_ref_zero);
                Warp_map[ref_ftr->id_] = ot;
            }

            for(int pyramid_level=0; pyramid_level<=2; pyramid_level++)
            {                
                warpAffine(A_cur_ref_zero, ref_ftr->img, ref_ftr->px, ref_ftr->level, search_level, pyramid_level, patch_size_half, patch_wrap.data(), ref_ftr->camera_id);
            }

            getpatch(img, pc, patch_cache.data(), 0, cam_id);

            if(ncc_en)
            {
                double ncc = NCC(patch_wrap.data(), patch_cache.data(), patch_size_total);
                if(ncc < ncc_thre) continue;
            }

            float error = 0.0;
            for (int ind=0; ind<patch_size_total; ind++) 
            {
                error += (patch_wrap[ind]-patch_cache[ind]) * (patch_wrap[ind]-patch_cache[ind]);
            }
            if(error > outlier_threshold*patch_size_total) continue;

            sub_map_cur_frame_.push_back(pt);
            sub_sparse_map->propa_errors.push_back(error);
            sub_sparse_map->search_levels.push_back(search_level);
            sub_sparse_map->errors.push_back(error);
            sub_sparse_map->index.push_back(i);  
            sub_sparse_map->voxel_points.push_back(pt);
            sub_sparse_map->patch.push_back(std::move(patch_wrap));
            // 多相机修改：记录该点来自哪个相机
            sub_sparse_map->camera_ids.push_back(cam_id);
        }
    }
    printf("[ VIO ][Camera %d]: choose %d points from sub_sparse_map.\n", cam_id, (int)sub_sparse_map->index.size());
}

void LidarSelector::addObservation(cv::Mat img)
{
    int total_points = (int)sub_sparse_map->index.size();
    if (total_points==0) return;

    for (int i=0; i<total_points; i++) 
    {
        PointPtr pt = sub_sparse_map->voxel_points[i];
        if(pt==nullptr) continue;
        int cam_id = sub_sparse_map->camera_ids[i];
        V2D pc(new_frame_->w2c(pt->pos_, cam_id));
        SE3 pose_cur = new_frame_->T_f_w_;
        bool add_flag = false;
        FeaturePtr last_feature = pt->obs_.back();
        SE3 pose_ref = last_feature->T_f_w_;
        SE3 delta_pose = pose_ref * pose_cur.inverse();
        double delta_p = delta_pose.translation().norm();
        double delta_theta = (delta_pose.rotation_matrix().trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (delta_pose.rotation_matrix().trace() - 1));            
        if(delta_p > 0.5 || delta_theta > 10) add_flag = true;

        Vector2d last_px = last_feature->px;
        double pixel_dist = (pc-last_px).norm();
        if(pixel_dist > 40) add_flag = true;
        if(pt->obs_.size()>=20)
        {
            FeaturePtr ref_ftr;
            pt->getFurthestViewObs(new_frame_->pos(), ref_ftr);
            pt->deleteFeatureRef(ref_ftr);
        }
        if(add_flag)
        {
            float val = vk::shiTomasiScore(img, pc[0], pc[1]);
            Vector3d f = cams_[cam_id]->cam2world(pc[0], pc[1]);
            FeaturePtr ftr_new(new Feature(pc, f, new_frame_->T_f_w_, val, sub_sparse_map->search_levels[i], cam_id)); 
            ftr_new->img = new_frame_->img_pyr_[cam_id][0];
            ftr_new->id_ = new_frame_->id_;
            pt->addFrameRef(ftr_new);      
        }
    }
}

// ComputeJ与UpdateState 需要根据camera_id选择对应的fx, fy, 宽高
float LidarSelector::UpdateState(cv::Mat img, float total_residual, int level)
{
    int total_points = (int)sub_sparse_map->index.size();
    if (total_points==0) return 0.;
    StatesGroup old_state = (*state);
    bool EKF_end = false;
    float error=0.0, last_error=total_residual;
    size_t n_meas_=0;

    // Z测量向量维度
    int H_DIM = total_points * patch_size_total;
    VectorXd z(H_DIM);
    z.setZero();

    H_sub.resize(H_DIM, 6);
    H_sub.setZero();

    for (int iteration=0; iteration<NUM_MAX_ITERATIONS; iteration++) 
    {
        error = 0.0;
        n_meas_ =0;
        M3D Rwi(state->rot_end);
        V3D Pwi(state->pos_end);
        Rcw = Rci * Rwi.transpose();
        Pcw = -Rci*Rwi.transpose()*Pwi + Pci;
        Jdp_dt = Rci * Rwi.transpose();

        // 遍历每个点，计算误差和雅可比
        for (int i=0; i<total_points; i++)
        {
            PointPtr pt = sub_sparse_map->voxel_points[i];
            if(pt==nullptr) continue;
            int search_level = sub_sparse_map->search_levels[i];
            int cam_id = sub_sparse_map->camera_ids[i];

            V3D pf = Rcw * pt->pos_ + Pcw;
            MD(2,3) Jdpi;
            dpi(pf, Jdpi);

            double fx = fxs_[cam_id], fy = fys_[cam_id];
            double cx = cxs_[cam_id], cy = cys_[cam_id];
            V2D pc;
            pc[0] = fx * pf[0]/pf[2] + cx;
            pc[1] = fy * pf[1]/pf[2] + cy;

            const int scale =  (1<<level)*(1<<search_level);
            const int u_ref_i = floorf(pc[0]/scale)*scale; 
            const int v_ref_i = floorf(pc[1]/scale)*scale;
            const float subpix_u_ref = (pc[0]-u_ref_i)/scale;
            const float subpix_v_ref = (pc[1]-v_ref_i)/scale;
            const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
            const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
            const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
            const float w_ref_br = subpix_u_ref * subpix_v_ref;

            // 对patch内像素计算残差和雅可比
            M3D p_hat;
            p_hat << SKEW_SYM_MATRX(pf);
            float patch_error=0.0;
            uint8_t* img_ptr_base = (uint8_t*) img.data + (v_ref_i-patch_size_half*scale)*widths_[cam_id] + (u_ref_i-patch_size_half*scale);

            for (int x=0; x<patch_size; x++) 
            {
                int row = (v_ref_i - patch_size_half*scale + x*scale);
                if(row<0 || row>=heights_[cam_id]-scale) continue;
                uint8_t* img_ptr_line = img_ptr_base + x*scale*widths_[cam_id];
                for (int y=0; y<patch_size; y++) 
                {
                    int col = (u_ref_i - patch_size_half*scale + y*scale);
                    if(col<0 || col>=widths_[cam_id]-scale)
                    {
                        z(i*patch_size_total+x*patch_size+y) = 0.0;
                        continue;
                    }
                    uint8_t* img_ptr_xy = img_ptr_line + y*scale;
                    float interp_val = w_ref_tl*img_ptr_xy[0] + 
                                       w_ref_tr*img_ptr_xy[scale] + 
                                       w_ref_bl*img_ptr_xy[scale*widths_[cam_id]] + 
                                       w_ref_br*img_ptr_xy[scale*widths_[cam_id]+scale];

                    float res = interp_val - sub_sparse_map->patch[i][patch_size_total*level + x*patch_size+y];
                    z(i*patch_size_total+x*patch_size+y) = res;
                    patch_error += res*res;
                    n_meas_++;

                    // 图像梯度近似
                    float du = 0.5f * ((w_ref_tl*img_ptr_xy[scale] + w_ref_tr*img_ptr_xy[2*scale] + w_ref_bl*img_ptr_xy[scale*widths_[cam_id]+scale] + w_ref_br*img_ptr_xy[scale*widths_[cam_id]+2*scale])
                                -(w_ref_tl*img_ptr_xy[-scale] + w_ref_tr*img_ptr_xy[0] + w_ref_bl*img_ptr_xy[scale*widths_[cam_id]-scale] + w_ref_br*img_ptr_xy[scale*widths_[cam_id]]));
                    float dv = 0.5f * ((w_ref_tl*img_ptr_xy[scale*widths_[cam_id]] + w_ref_tr*img_ptr_xy[scale+scale*widths_[cam_id]] + w_ref_bl*img_ptr_xy[2*scale*widths_[cam_id]] + w_ref_br*img_ptr_xy[2*scale*widths_[cam_id]+scale])
                                -(w_ref_tl*img_ptr_xy[-scale*widths_[cam_id]] + w_ref_tr*img_ptr_xy[-scale*widths_[cam_id]+scale] + w_ref_bl*img_ptr_xy[0] + w_ref_br*img_ptr_xy[scale]));
                    du /= scale;
                    dv /= scale;

                    MD(1,2) Jimg; 
                    Jimg << du, dv;
                    // 将相机内参fx, fy考虑进来
                    // dpi计算在后面的JdR,Jdt中反映
                    // 此处可根据误差更精确方式计算，在此略
                    M3D Rwi(state->rot_end);
                    MD(1,3) Jdphi, Jdp;
                    Jdphi = Jimg * Jdpi * p_hat;
                    Jdp = - Jimg * Jdpi;

                    MD(1,6) JH;
                    JH.block<1,3>(0,0) = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
                    JH.block<1,3>(0,3) = Jdp * Jdp_dt;
                    H_sub.block<1,6>(i*patch_size_total+x*patch_size+y,0) = JH;
                }
            }
            sub_sparse_map->errors[i] = patch_error;
            error += patch_error;
        }

        error = error/n_meas_;
        if (error <= last_error) 
        {
            old_state = (*state);
            last_error = error;

            auto H_sub_T = H_sub.transpose();
            H_T_H.block<6,6>(0,0) = H_sub_T * H_sub;
            Matrix<double,DIM_STATE,DIM_STATE> K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
            auto HTz = H_sub_T * z;
            G.block<DIM_STATE,6>(0,0) = K_1.block<DIM_STATE,6>(0,0)*H_T_H.block<6,6>(0,0);
            auto vec = (*state_propagat) - (*state);
            auto solution = -K_1.block<DIM_STATE,6>(0,0)*HTz + vec - G.block<DIM_STATE,6>(0,0)*vec.block<6,1>(0,0);
            (*state) += solution;
            auto rot_add = solution.block<3,1>(0,0);
            auto t_add   = solution.block<3,1>(3,0);
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

        if (iteration==NUM_MAX_ITERATIONS || EKF_end) 
        {
            break;
        }
    }
    return last_error;
}

void LidarSelector::ComputeJ(cv::Mat img)
{
    int total_points = (int)sub_sparse_map->index.size();
    if (total_points==0) return;
    float error = 1e10;
    float now_error = error;
    for (int level=2; level>=0; level--) 
    {
        now_error = UpdateState(img, error, level);
    }
    if (now_error < error)
    {
        state->cov -= G*state->cov;
    }
    updateFrameState(*state);
}

void LidarSelector::updateFrameState(StatesGroup state)
{
    M3D Rwi(state.rot_end);
    V3D Pwi(state.pos_end);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci*Rwi.transpose()*Pwi + Pci;
    new_frame_->T_f_w_ = SE3(Rcw, Pcw);
}

V3F LidarSelector::getpixel(cv::Mat img, V2D pc, int cam_id) 
{
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0]); 
    const int v_ref_i = floorf(pc[1]);
    const float subpix_u_ref = (u_ref-u_ref_i);
    const float subpix_v_ref = (v_ref-v_ref_i);
    const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
    const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;

    if(u_ref_i<0||u_ref_i>=widths_[cam_id]-1 || v_ref_i<0||v_ref_i>=heights_[cam_id]-1)
        return V3F(0,0,0);

    uint8_t* img_ptr = (uint8_t*) img.data + ((v_ref_i)*widths_[cam_id] + (u_ref_i))*3;
    float B = w_ref_tl*img_ptr[0] + w_ref_tr*img_ptr[3] + w_ref_bl*img_ptr[widths_[cam_id]*3] + w_ref_br*img_ptr[widths_[cam_id]*3+3];
    float G = w_ref_tl*img_ptr[1] + w_ref_tr*img_ptr[4] + w_ref_bl*img_ptr[widths_[cam_id]*3+1] + w_ref_br*img_ptr[widths_[cam_id]*3+4];
    float R = w_ref_tl*img_ptr[2] + w_ref_tr*img_ptr[5] + w_ref_bl*img_ptr[widths_[cam_id]*3+2] + w_ref_br*img_ptr[widths_[cam_id]*3+5];
    return V3F(B,G,R);
}

// 多相机扩展detect：现在传入多个图像imgs，每个相机一张
void LidarSelector::detect(const std::vector<cv::Mat>& imgs, pcl::PointCloud<PointType>::Ptr pg)
{
    // 假定所有imgs与cams_对应一致
    if(imgs.size() != cams_.size())
    {
        std::cout << "Error: Number of images does not match number of cameras." << std::endl;
        return;
    }

    img_rgb = imgs[0].clone();
    img_cp = imgs[0].clone();
    cv::Mat img_gray;
    cv::cvtColor(imgs[0], img_gray, CV_BGR2GRAY);

    // 初始化new_frame_
    new_frame_.reset(new Frame(cams_, imgs));
    updateFrameState(*state);

    if(stage_ == STAGE_FIRST_FRAME && pg->size()>10)
    {
        new_frame_->setKeyframe();
        stage_ = STAGE_DEFAULT_FRAME;
    }

    double t1 = omp_get_wtime();

    // 对每个相机执行addFromSparseMap和addSparseMap
    // 将结果统一存入sub_sparse_map
    for(size_t cam_id = 0; cam_id < cams_.size(); cam_id++)
    {
        cv::Mat img_mono;
        if(imgs[cam_id].channels()>1)
            cv::cvtColor(imgs[cam_id], img_mono, CV_BGR2GRAY);
        else
            img_mono = imgs[cam_id];

        addFromSparseMap(img_mono, pg, (int)cam_id);
        addSparseMap(img_mono, pg, (int)cam_id);
    }

    double t4 = omp_get_wtime();
    
    ComputeJ(img_gray);

    double t5 = omp_get_wtime();

    addObservation(img_gray);
    
    double t2 = omp_get_wtime();
    frame_count ++;
    ave_total = ave_total * (frame_count - 1) / frame_count + (t2 - t1) / frame_count;

    printf("[ VIO ]: time: total: %.6f ave_total: %.6f.\n", t2-t1, ave_total);

    // 可根据需要显示特征点
    display_keypatch(t2-t1);
}

void LidarSelector::display_keypatch(double time)
{
    int total_points = (int)sub_sparse_map->index.size();
    if (total_points==0) return;
    // 默认显示相机0的结果，如果需要对多个相机做可视化请自行扩展
    for(int i=0; i<total_points; i++)
    {
        PointPtr pt = sub_sparse_map->voxel_points[i];
        int cam_id = sub_sparse_map->camera_ids[i];
        V2D pc(new_frame_->w2c(pt->pos_, cam_id));
        cv::Point2f pf(pc[0], pc[1]); 
        if (sub_sparse_map->errors[i]<8000)
            cv::circle(img_cp, pf, 6, cv::Scalar(0, 255, 0), -1, 8);
        else
            cv::circle(img_cp, pf, 6, cv::Scalar(255, 0, 0), -1, 8);
    }   
    std::string text = std::to_string(int(1/time))+" HZ";
    cv::Point2f origin(20,20);
    cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, 8, 0);
}

} // namespace lidar_selection
