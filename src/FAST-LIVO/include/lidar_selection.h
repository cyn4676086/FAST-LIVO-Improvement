#ifndef LIDAR_SELECTION_H_
#define LIDAR_SELECTION_H_

#include <common_lib.h>
#include <vikit/abstract_camera.h>
#include <frame.h>
#include <map.h>
#include <feature.h>
#include <point.h>
#include <vikit/vision.h>
#include <vikit/math_utils.h>
#include <vikit/robust_cost.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <set>
#include <vector>
#include <memory>

namespace lidar_selection {

struct CameraInfo {
    vk::AbstractCamera* cam;  // 指向相机对象的指针
    int cam_id;

    // 内参
    double fx;
    double fy;
    double cx;
    double cy;

    // 外参
    M3D Rcl;  // 相机旋转矩阵到雷达
    V3D Pcl;  // 相机平移向量到雷达
    M3D Rci;  // 相机到 IMU 的旋转矩阵
    V3D Pci;  // 相机到 IMU 的平移向量
    M3D Rcw;  // 相机到世界的旋转矩阵
    V3D Pcw;  // 相机到世界的平移向量

    // 雅可比矩阵
    M3D Jdphi_dR;  // 姿态变化对旋转矩阵的雅可比矩阵
    M3D Jdp_dR;    // 位置变化对旋转矩阵的雅可比矩阵
    V3D Jdp_dt;    // 位置变化对平移向量的雅可比矩阵

    // 图像尺寸
    int width;
    int height;
};


class LidarSelector {
  //多个相机 一个地图
  public:
    int grid_size;
    std::vector<CameraInfo> cameras;  // 支持多个相机
    SparseMap* sparse_map;
    StatesGroup* state;
    StatesGroup* state_propagat;
    
    M3D Rli; 
    V3D Pli; 
    int* align_flag;
    int* grid_num;
    int* map_index;
    float* map_dist;
    float* map_value;
    float* patch_with_border_;
    vector<float> patch_cache;
    int width,height,grid_n_width, grid_n_height, length;
    SubSparseMap* sub_sparse_map;

    bool ncc_en;
    int debug, patch_size, patch_size_total, patch_size_half;
    int MIN_IMG_COUNT;
    int NUM_MAX_ITERATIONS;
    vk::robust_cost::WeightFunctionPtr weight_function_;
    float weight_scale_;
    double img_point_cov, outlier_threshold, ncc_thre;
    size_t n_meas_;                //!< Number of measurements
    deque< PointPtr > map_cur_frame_;
    deque< PointPtr > sub_map_cur_frame_;
    double computeH, ekf_time;
    double ave_total = 0.0;
    int frame_count = 0;
    vk::robust_cost::ScaleEstimatorPtr scale_estimator_;

    Matrix<double, DIM_STATE, DIM_STATE> G, H_T_H;
    MatrixXd H_sub, K;
    cv::flann::Index Kdtree;

     // 构造函数：接受网格大小、稀疏地图指针和多个相机指针
    LidarSelector(const int grid_size, SparseMap* sparse_map, const std::vector<vk::AbstractCamera*>& camera_ptrs);

    ~LidarSelector();

    // 修改后的函数以处理多个相机
    void detect(const std::vector<cv::Mat>& imgs, PointCloudXYZI::Ptr pg);
    float CheckGoodPoints(cv::Mat img, V2D uv);
    void addFromSparseMap(const std::vector<cv::Mat>& imgs, PointCloudXYZI::Ptr pg);
    void addSparseMap(const std::vector<cv::Mat>& imgs, PointCloudXYZI::Ptr pg);
    void FeatureAlignment(const std::vector<cv::Mat>& imgs);
    void set_extrinsic(const V3D &transl, const M3D &rot); // 不作修改
    void set_camera2lidar(const std::vector<double>& R, const std::vector<double>& P)
    void init();
    void getpatch(cv::Mat img, V3D pg, float* patch_tmp, int level);
    void dpi(V3D p, MD(2,3)& J);
    float UpdateState(const std::vector<cv::Mat>& imgs, float total_residual, int level);
    double NCC(float* ref_patch, float* cur_patch, int patch_size);

    void ComputeJ(const std::vector<cv::Mat>& imgs);
    void reset_grid();
    void addObservation(const std::vector<cv::Mat>& imgs);
    void reset();
    bool initialization(FramePtr cur_frame, PointCloudXYZI::Ptr pg);   
    void createPatchFromPatchWithBorder(float* patch_with_border, float* patch_ref);
    void getWarpMatrixAffine(
      const vk::AbstractCamera& cam,
      const Vector2d& px_ref,
      const Vector3d& f_ref,
      const double depth_ref,
      const SE3& T_cur_ref,
      const int level_ref,    // px_ref对应特征点的金字塔层级
      const int pyramid_level,
      const int halfpatch_size,
      Matrix2d& A_cur_ref);
    bool align2D(
      const cv::Mat& cur_img,
      float* ref_patch_with_border,
      float* ref_patch,
      const int n_iter,
      Vector2d& cur_px_estimate,
      int index);
    void AddPoint(PointPtr pt_new);
    int getBestSearchLevel(const Matrix2d& A_cur_ref, const int max_level);
    void display_keypatch(double time);
    void updateFrameState(StatesGroup state);
    V3F getpixel(cv::Mat img, V2D pc);

    void warpAffine(
      const Matrix2d& A_cur_ref,
      const cv::Mat& img_ref,
      const Vector2d& px_ref,
      const int level_ref,
      const int search_level,
      const int pyramid_level,
      const int halfpatch_size,
      float* patch);
    
    PointCloudXYZI::Ptr Map_points;
    PointCloudXYZI::Ptr Map_points_output;
    PointCloudXYZI::Ptr pg_down;
    pcl::VoxelGrid<PointType> downSizeFilter;
    std::unordered_map<VOXEL_KEY, VOXEL_POINTS*, VOXEL_KEY_Hash> feat_map;
    std::unordered_map<VOXEL_KEY, float, VOXEL_KEY_Hash> sub_feat_map; // timestamp
    std::unordered_map<int, Warp*> Warp_map; // reference frame id, A_cur_ref and search_level

    std::vector<VOXEL_KEY> occupy_postions;
    std::set<VOXEL_KEY> sub_postion;
    std::vector<PointPtr> voxel_points_;
    std::vector<V3D> add_voxel_points_;

    cv::Mat img_cp, img_rgb;//还需修改
    std::vector<FramePtr> overlap_kfs_;
    FramePtr new_frame_;
    FramePtr last_kf_;
    Map map_;
    enum Stage {
      STAGE_FIRST_FRAME,
      STAGE_DEFAULT_FRAME
    };
    Stage stage_;
    enum CellType {
      TYPE_MAP = 1,
      TYPE_POINTCLOUD,
      TYPE_UNKNOWN
    };

private:
    struct Candidate 
    {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      PointPtr pt;       
      Vector2d px;    
      Candidate(PointPtr pt, Vector2d& px) : pt(pt), px(px) {}
    };
    typedef std::list<Candidate > Cell;
    typedef std::vector<Cell*> CandidateGrid;

    /// The grid stores a set of candidate matches. For every grid cell we try to find one match.
    struct Grid
    {
      CandidateGrid cells;
      std::vector<int> cell_order;
      int cell_size;
      int grid_n_cols;
      int grid_n_rows;
      int cell_length;
    };

    Grid grid_;
};

typedef boost::shared_ptr<LidarSelector> LidarSelectorPtr;

} // namespace lidar_selection

#endif // LIDAR_SELECTION_H_