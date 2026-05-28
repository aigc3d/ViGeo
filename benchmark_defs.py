TASK_MONO_DEPTH = 'mono_depth'
TASK_VIDEO_DEPTH = 'video_depth'
TASK_POINTMAP = 'pointmap'
TASK_NORMAL = 'normal'

VIGEO_TASKS = [TASK_MONO_DEPTH, TASK_VIDEO_DEPTH, TASK_POINTMAP, TASK_NORMAL]
DEPTH_BENCHMARK_TASKS = [TASK_VIDEO_DEPTH, TASK_MONO_DEPTH, TASK_POINTMAP]

DEPTH_METRICS = ['absrel', 'd1']
NORMAL_METRICS = ['mean', 'median', 'a3']

DEPTH_SUMMARY_COLUMNS = ['task', 'dataset', 'benchmark', *DEPTH_METRICS]
NORMAL_SUMMARY_COLUMNS = ['task', 'dataset', 'benchmark', *NORMAL_METRICS]
VIGEO_SUMMARY_COLUMNS = ['task', 'dataset', 'benchmark', *DEPTH_METRICS, *NORMAL_METRICS]

VIGEO_DEFAULT_DATASETS = {
    TASK_MONO_DEPTH: ['sintel', 'bonn', 'kitti'],
    TASK_VIDEO_DEPTH: ['sintel', 'bonn', 'kitti'],
    TASK_POINTMAP: ['sintel', 'bonn', 'kitti'],
    TASK_NORMAL: ['sintel', 'nyuv2', 'hammer'],
}

VIGEO_SUPPORTED_DATASETS = {
    TASK_MONO_DEPTH: ['sintel', 'bonn', 'kitti', 'bonn_400', 'kitti_300', 'hammer'],
    TASK_VIDEO_DEPTH: ['sintel', 'bonn', 'kitti', 'bonn_400', 'kitti_300', 'hammer'],
    TASK_POINTMAP: ['sintel', 'bonn', 'kitti'],
    TASK_NORMAL: ['sintel', 'nyuv2', 'hammer'],
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
