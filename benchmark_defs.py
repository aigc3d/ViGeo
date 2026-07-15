TASK_MONO_DEPTH = 'mono_depth'
TASK_VIDEO_DEPTH = 'video_depth'
TASK_POINTMAP = 'pointmap'
TASK_NORMAL = 'normal'
TASK_POSE_ESTIMATION = 'pose_estimation'
TASK_RECONSTRUCTION = 'reconstruction'

VIGEO_TASKS = [TASK_MONO_DEPTH, TASK_VIDEO_DEPTH, TASK_POINTMAP, TASK_NORMAL, TASK_POSE_ESTIMATION, TASK_RECONSTRUCTION]
DEPTH_BENCHMARK_TASKS = [TASK_VIDEO_DEPTH, TASK_MONO_DEPTH, TASK_POINTMAP]

DEPTH_METRICS = ['absrel', 'd1']
NORMAL_METRICS = ['mean', 'median', 'a3']
POSE_METRICS = ['ate', 'rpe_trans', 'rpe_rot']
RECONSTRUCTION_METRICS = ['acc_mean', 'acc_median', 'comp_mean', 'comp_median', 'nc_mean', 'nc_median']

DEPTH_SUMMARY_COLUMNS = ['task', 'dataset', 'benchmark', *DEPTH_METRICS]
NORMAL_SUMMARY_COLUMNS = ['task', 'dataset', 'benchmark', *NORMAL_METRICS]
VIGEO_SUMMARY_COLUMNS = [
    'task',
    'dataset',
    'benchmark',
    'num_sequences',
    'num_failed',
    'num_scenes',
    *DEPTH_METRICS,
    *NORMAL_METRICS,
    *POSE_METRICS,
    *RECONSTRUCTION_METRICS,
]

VIGEO_DEFAULT_DATASETS = {
    TASK_MONO_DEPTH: ['sintel', 'bonn', 'kitti'],
    TASK_VIDEO_DEPTH: ['sintel', 'bonn', 'kitti'],
    TASK_POINTMAP: ['sintel', 'bonn', 'kitti'],
    TASK_NORMAL: ['sintel', 'nyuv2', 'hammer'],
    TASK_POSE_ESTIMATION: ['sintel'],
    TASK_RECONSTRUCTION: ['7scenes', 'nrgbd'],
}

VIGEO_SUPPORTED_DATASETS = {
    TASK_MONO_DEPTH: ['sintel', 'bonn', 'kitti', 'bonn_400', 'kitti_300', 'hammer'],
    TASK_VIDEO_DEPTH: ['sintel', 'bonn', 'kitti', 'bonn_400', 'kitti_300', 'hammer'],
    TASK_POINTMAP: ['sintel', 'bonn', 'kitti'],
    TASK_NORMAL: ['sintel', 'nyuv2', 'hammer'],
    TASK_POSE_ESTIMATION: ['sintel'],
    TASK_RECONSTRUCTION: ['7scenes', '7scenes-sparse', 'nrgbd', 'nrgbd-sparse'],
}

DEPTH_BENCHMARK_DEFAULT_DATASETS = {
    TASK_VIDEO_DEPTH: ['sintel', 'bonn', 'kitti'],
    TASK_MONO_DEPTH: ['sintel', 'bonn', 'kitti'],
    TASK_POINTMAP: ['sintel', 'bonn', 'kitti'],
}

DEPTH_BENCHMARK_SUPPORTED_DATASETS = {
    TASK_VIDEO_DEPTH: ['sintel', 'bonn', 'kitti', 'bonn_200', 'bonn_400', 'kitti_300', 'hammer'],
    TASK_MONO_DEPTH: ['sintel', 'bonn', 'kitti', 'bonn_200', 'bonn_400', 'kitti_300', 'hammer'],
    TASK_POINTMAP: ['sintel', 'bonn', 'kitti'],
}

NORMAL_DATASETS = ['hammer', 'sintel', 'nyuv2']
