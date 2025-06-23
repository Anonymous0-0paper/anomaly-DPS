import torch
from config import MODEL_CONFIG, DATA_CONFIG, EXPERIMENT_CONFIG
from data_loader import get_data_loader
from model import AnomalyDetectionPipeline
from evaluator import Evaluator
import pandas as pd
import os
import time
import numpy as np
import logging
import sys
import gc

# Configure logging system
def setup_logger():
    logger = logging.getLogger()
    log_level = EXPERIMENT_CONFIG.get('log_level', 'INFO').upper()
    logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler(
        f"logs/experiment_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    return logger

def run_experiment(logger, dataset_type, dataset_params):
    """
    Run experiment
    Process specified dataset according to configuration
    Parameters:
    logger: Logger
    dataset_type: Dataset type ('nasa', 'smd', 'swat', 'wadi')
    dataset_params: Dataset parameters
    Returns:
    results: List of results
    """
    logger.info(f"Starting experiment for {dataset_type} dataset")

    # Get data loader
    try:
        # NASA dataset requires additional dataset parameter
        if dataset_type == 'nasa':
            loader = get_data_loader(
                dataset_type=dataset_type,
                root_dir=dataset_params['root_dir'],
                config=dataset_params,
                dataset=dataset_params.get('dataset', 'MSL')
            )
        else:
            loader = get_data_loader(
                dataset_type=dataset_type,
                root_dir=dataset_params['root_dir'],
                config=dataset_params
            )
    except Exception as e:
        logger.error(f"Failed to initialize data loader: {str(e)}")
        return []

    # Get all items
    all_items = loader.get_all_items()

    # Determine the list of items to process
    if dataset_params.get('test_items'):  # General processing
        items = [i for i in dataset_params['test_items'] if i in all_items]
    else:
        # Process all items (possibly excluding some items)
        exclude_items = dataset_params.get('exclude_items', [])
        items = [i for i in all_items if i not in exclude_items]

    # Limit maximum number of items
    max_items = dataset_params.get('max_items')
    if max_items is not None and max_items > 0:
        items = items[:max_items]

    logger.info(f"Found {len(all_items)} items in {dataset_type} dataset")
    logger.info(f"Processing {len(items)} items: {', '.join(items[:5])}{'...' if len(items) > 5 else ''}")

    # Initialize evaluator
    evaluator = Evaluator(
        save_plots=EXPERIMENT_CONFIG['save_plots'],
        plot_dir=EXPERIMENT_CONFIG['plot_dir'],
        max_plot_points=EXPERIMENT_CONFIG.get('max_plot_points', 5000)
    )

    results = []

    # Get sampling configuration
    sampling_config = EXPERIMENT_CONFIG['data_sampling']
    enable_sampling = sampling_config.get('enable', False)
    logger.info(f"Data sampling {'ENABLED' if enable_sampling else 'DISABLED'}")

    # Process all selected items
    for i, item_id in enumerate(items):
        logger.info(f"\nProcessing item {i + 1}/{len(items)}: {item_id}")

        try:
            # Load data
            train_data, test_data, test_labels = loader.load_item(item_id)

            # Check data validity
            if len(train_data) == 0 or len(test_data) == 0:
                raise ValueError("Empty data loaded")

            # Record original data size
            orig_train_size = len(train_data)
            orig_test_size = len(test_data)
            logger.info(f"Original data sizes - Train: {orig_train_size}, Test: {orig_test_size}")

            # ================= New: Feature Standardization =================
            logger.info("Applying feature scaling...")
            from sklearn.preprocessing import StandardScaler

            # Note: Fit scaler only using training data
            scaler = StandardScaler()
            train_data = scaler.fit_transform(train_data)
            test_data = scaler.transform(test_data)

            # Data sampling (quick validation mode)
            if enable_sampling:
                logger.info(f"Applying data sampling for {item_id}")

                # Training data sampling
                train_sample_ratio = sampling_config.get('train_sample_ratio', 0.1)
                max_train_points = sampling_config.get('max_train_points', 5000)
                if train_sample_ratio < 1.0:
                    n_train = min(int(len(train_data) * train_sample_ratio), max_train_points)
                    indices = np.random.choice(len(train_data), n_train, replace=False)
                    train_data = train_data[indices]
                    logger.info(f"Sampled training data: {len(train_data)}/{orig_train_size} points")
                else:
                    logger.info("Using full training data")

                # Test data sampling
                test_sample_ratio = sampling_config.get('test_sample_ratio', 0.1)
                max_test_points = sampling_config.get('max_test_points', 10000)
                if test_sample_ratio < 1.0:
                    n_test = min(int(len(test_data) * test_sample_ratio), max_test_points)
                    indices = np.random.choice(len(test_data), n_test, replace=False)
                    test_data = test_data[indices]
                    test_labels = test_labels[indices]
                    logger.info(f"Sampled test data: {len(test_data)}/{orig_test_size} points")
                else:
                    logger.info("Using full test data")

            # Initialize pipeline
            pipeline = AnomalyDetectionPipeline(
                model_params=MODEL_CONFIG,
                device=EXPERIMENT_CONFIG['device']
            )

            # Training and evaluation
            logger.info("Starting training and evaluation...")
            scores, aligned_labels = pipeline.fit(train_data, test_data, test_labels, logger)
            logger.info("Training and evaluation completed")

            # Calculate metrics
            metrics = evaluator.calculate_metrics(scores, aligned_labels)

            # Store results
            result = {
                'dataset': dataset_type,
                'item': item_id,
                'orig_train_samples': orig_train_size,
                'orig_test_samples': orig_test_size,
                'used_train_samples': len(train_data),
                'used_test_samples': len(test_data),
                'sampling_enabled': enable_sampling,
                'sampling_ratio': f"{train_sample_ratio:.2f}/{test_sample_ratio:.2f}" if enable_sampling else "N/A",
                'features': train_data.shape[1],
                **metrics
            }
            results.append(result)
            logger.info(f"Results for {item_id}: F1={metrics['f1']:.4f}, AUC={metrics['auc_roc']:.4f}")

            # Visualize results
            try:
                plot_success = evaluator.plot_results(
                    test_data, scores, aligned_labels,
                    metrics['threshold'], f"{dataset_type}_{item_id}"
                )
                if not plot_success:
                    logger.warning(f"Plotting failed for {item_id}")
            except Exception as e:
                logger.error(f"Error during plotting for {item_id}: {str(e)}")

            # Clean up memory
            del train_data, test_data, test_labels, scores, aligned_labels
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing item {item_id}: {str(e)}")
            results.append({
                'dataset': dataset_type,
                'item': item_id,
                'error': str(e),
                'f1': np.nan,
                'auc_roc': np.nan,
                'threshold': np.nan
            })

    return results

def main():
    """Main program"""
    # Set up logging
    logger = setup_logger()
    logger.info("Starting anomaly detection experiment")
    logger.info(f"Device: {EXPERIMENT_CONFIG['device']}")
    logger.info(f"Model config: {MODEL_CONFIG}")

    # Ensure results directory exists
    os.makedirs(os.path.dirname(EXPERIMENT_CONFIG['results_file']), exist_ok=True)
    os.makedirs(EXPERIMENT_CONFIG['plot_dir'], exist_ok=True)

    start_time = time.time()
    all_results = []

    # Iterate through all dataset configurations
    for dataset_config in DATA_CONFIG:
        dataset_name = dataset_config['name']
        dataset_type = dataset_config['type']
        dataset_params = dataset_config['params']

        logger.info("\n" + "=" * 50)
        logger.info(f"Processing {dataset_name} Dataset")
        logger.info("=" * 50)
        logger.info(f"Dataset type: {dataset_type}")

        # Apply dataset-specific sampling configuration
        sampling_config = EXPERIMENT_CONFIG['data_sampling'].copy()
        if 'dataset_specific_sampling' in EXPERIMENT_CONFIG:
            if dataset_name in EXPERIMENT_CONFIG['dataset_specific_sampling']:
                # Merge configurations
                sampling_config.update(
                    EXPERIMENT_CONFIG['dataset_specific_sampling'][dataset_name]
                )

        # Save original sampling configuration
        original_sampling = EXPERIMENT_CONFIG['data_sampling']

        # Temporarily update sampling configuration
        EXPERIMENT_CONFIG['data_sampling'] = sampling_config

        try:
            # Run experiment
            results = run_experiment(logger, dataset_type, dataset_params)

            # Add dataset name to results
            for res in results:
                res['dataset_name'] = dataset_name
                res['dataset_type'] = dataset_type

            all_results.extend(results)

            # Save intermediate results
            interim_df = pd.DataFrame(all_results)
            interim_file = EXPERIMENT_CONFIG['results_file'].replace('.csv', '_interim.csv')
            interim_df.to_csv(interim_file, index=False)
            logger.info(f"Interim results saved to {interim_file}")

        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {str(e)}")
            all_results.append({
                'dataset_name': dataset_name,
                'dataset_type': dataset_type,
                'error': str(e)
            })

        finally:
            # Restore original sampling configuration
            EXPERIMENT_CONFIG['data_sampling'] = original_sampling

            # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save final results
    results_df = pd.DataFrame(all_results)
    results_file = EXPERIMENT_CONFIG['results_file']
    results_df.to_csv(results_file, index=False)
    logger.info(f"Final results saved to {results_file}")

    # Print summary results
    logger.info("\n" + "=" * 50)
    logger.info("Summary of All Results")
    logger.info("=" * 50)

    # Group summary by dataset
    for dataset_name, group in results_df.groupby('dataset_name'):
        valid_results = group.dropna(subset=['f1'])
        failed_count = len(group) - len(valid_results)
        logger.info(f"\n{dataset_name}:")
        logger.info(f"  Items processed: {len(group)}")
        logger.info(f"  Successful items: {len(valid_results)}")
        logger.info(f"  Failed items: {failed_count}")

        if not valid_results.empty:
            avg_f1 = valid_results['f1'].mean()
            avg_auc = valid_results['auc_roc'].mean()
            logger.info(f"  Average F1: {avg_f1:.4f}")
            logger.info(f"  Average AUC-ROC: {avg_auc:.4f}")

            # F1 score distribution
            logger.info(f"  F1 distribution:")
            logger.info(f"    Min: {valid_results['f1'].min():.4f}")
            logger.info(f"    25%: {valid_results['f1'].quantile(0.25):.4f}")
            logger.info(f"    Median: {valid_results['f1'].median():.4f}")
            logger.info(f"    75%: {valid_results['f1'].quantile(0.75):.4f}")
            logger.info(f"    Max: {valid_results['f1'].max():.4f}")

    # Display failed channels
    failed_results = results_df[results_df['f1'].isna()]
    if not failed_results.empty:
        logger.info("\nFailed items summary:")
        for _, row in failed_results.iterrows():
            logger.info(f"- {row['dataset_name']}: {row['item']} - {row.get('error', 'Unknown error')}")

    # Display total time taken
    elapsed = time.time() - start_time
    logger.info(f"\nTotal processing time: {elapsed / 60:.2f} minutes")

if __name__ == "__main__":
    main()


# import torch
# from config import MODEL_CONFIG, DATA_CONFIG, EXPERIMENT_CONFIG
# from data_loader import get_data_loader
# from model import AnomalyDetectionPipeline
# from evaluator import Evaluator
# import pandas as pd
# import os
# import time
# import numpy as np
# import logging
# import sys
# import gc
#
#
# # 配置日志系统
# def setup_logger():
#     logger = logging.getLogger()
#     log_level = EXPERIMENT_CONFIG.get('log_level', 'INFO').upper()
#     logger.setLevel(log_level)
#
#     # 控制台处理器
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_formatter = logging.Formatter(
#         '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
#     console_handler.setFormatter(console_formatter)
#     logger.addHandler(console_handler)
#
#     # 文件处理器
#     os.makedirs('logs', exist_ok=True)
#     file_handler = logging.FileHandler(
#         f"logs/experiment_{time.strftime('%Y%m%d_%H%M%S')}.log"
#     )
#     file_handler.setFormatter(console_formatter)
#     logger.addHandler(file_handler)
#
#     return logger
#
#
# def run_experiment(logger, dataset_type, dataset_params):
#     """
#     运行实验
#     根据配置处理指定数据集
#
#     参数:
#     logger: 日志记录器
#     dataset_type: 数据集类型 ('nasa', 'smd', 'swat', 'wadi')
#     dataset_params: 数据集参数
#
#     返回:
#     results: 结果列表
#     """
#     logger.info(f"Starting experiment for {dataset_type} dataset")
#
#     # 获取数据加载器
#     try:
#         # NASA 数据集需要额外传递 dataset 参数
#         if dataset_type == 'nasa':
#             loader = get_data_loader(
#                 dataset_type=dataset_type,
#                 root_dir=dataset_params['root_dir'],
#                 config=dataset_params,
#                 dataset=dataset_params.get('dataset', 'MSL')
#             )
#         else:
#             loader = get_data_loader(
#                 dataset_type=dataset_type,
#                 root_dir=dataset_params['root_dir'],
#                 config=dataset_params
#             )
#     except Exception as e:
#         logger.error(f"Failed to initialize data loader: {str(e)}")
#         return []
#
#     # 获取所有项目
#     all_items = loader.get_all_items()
#
#     # 确定要处理的项目列表
#     if dataset_params.get('test_items'):  # 通用处理
#         items = [i for i in dataset_params['test_items'] if i in all_items]
#     else:
#         # 处理所有项目（可能排除某些项目）
#         exclude_items = dataset_params.get('exclude_items', [])
#         items = [i for i in all_items if i not in exclude_items]
#
#     # 限制最大项目数
#     max_items = dataset_params.get('max_items')
#     if max_items is not None and max_items > 0:
#         items = items[:max_items]
#
#     logger.info(f"Found {len(all_items)} items in {dataset_type} dataset")
#     logger.info(f"Processing {len(items)} items: {', '.join(items[:5])}{'...' if len(items) > 5 else ''}")
#
#     # 初始化评估器
#     evaluator = Evaluator(
#         save_plots=EXPERIMENT_CONFIG['save_plots'],
#         plot_dir=EXPERIMENT_CONFIG['plot_dir'],
#         max_plot_points=EXPERIMENT_CONFIG.get('max_plot_points', 5000)
#     )
#
#     results = []
#
#     # 获取采样配置
#     sampling_config = EXPERIMENT_CONFIG['data_sampling']
#     enable_sampling = sampling_config.get('enable', False)
#     logger.info(f"Data sampling {'ENABLED' if enable_sampling else 'DISABLED'}")
#
#     # 处理所有选定项目
#     for i, item_id in enumerate(items):
#         logger.info(f"\nProcessing item {i + 1}/{len(items)}: {item_id}")
#
#         try:
#             # 加载数据
#             train_data, test_data, test_labels = loader.load_item(item_id)
#
#             # 检查数据有效性
#             if len(train_data) == 0 or len(test_data) == 0:
#                 raise ValueError("Empty data loaded")
#
#             # 原始数据大小记录
#             orig_train_size = len(train_data)
#             orig_test_size = len(test_data)
#             logger.info(f"Original data sizes - Train: {orig_train_size}, Test: {orig_test_size}")
#
#             # ================= 新增：特征标准化 =================
#             logger.info("Applying feature scaling...")
#             from sklearn.preprocessing import StandardScaler
#
#             # 注意：仅使用训练数据拟合缩放器
#             scaler = StandardScaler()
#             train_data = scaler.fit_transform(train_data)
#             test_data = scaler.transform(test_data)
#
#             # 数据采样 (快速验证模式)
#             if enable_sampling:
#                 logger.info(f"Applying data sampling for {item_id}")
#
#                 # 训练数据采样
#                 train_sample_ratio = sampling_config.get('train_sample_ratio', 0.1)
#                 max_train_points = sampling_config.get('max_train_points', 5000)
#                 if train_sample_ratio < 1.0:
#                     n_train = min(int(len(train_data) * train_sample_ratio), max_train_points)
#                     indices = np.random.choice(len(train_data), n_train, replace=False)
#                     train_data = train_data[indices]
#                     logger.info(f"Sampled training data: {len(train_data)}/{orig_train_size} points")
#                 else:
#                     logger.info("Using full training data")
#
#                 # 测试数据采样
#                 test_sample_ratio = sampling_config.get('test_sample_ratio', 0.1)
#                 max_test_points = sampling_config.get('max_test_points', 10000)
#                 if test_sample_ratio < 1.0:
#                     n_test = min(int(len(test_data) * test_sample_ratio), max_test_points)
#                     indices = np.random.choice(len(test_data), n_test, replace=False)
#                     test_data = test_data[indices]
#                     test_labels = test_labels[indices]
#                     logger.info(f"Sampled test data: {len(test_data)}/{orig_test_size} points")
#                 else:
#                     logger.info("Using full test data")
#
#             # 初始化管道
#             pipeline = AnomalyDetectionPipeline(
#                 model_params=MODEL_CONFIG,
#                 device=EXPERIMENT_CONFIG['device']
#             )
#
#             # 训练和评估
#             logger.info("Starting training and evaluation...")
#             scores, aligned_labels = pipeline.fit(train_data, test_data, test_labels, logger)
#             logger.info("Training and evaluation completed")
#
#             # 计算指标
#             metrics = evaluator.calculate_metrics(scores, aligned_labels)
#
#             # 存储结果
#             result = {
#                 'dataset': dataset_type,
#                 'item': item_id,
#                 'orig_train_samples': orig_train_size,
#                 'orig_test_samples': orig_test_size,
#                 'used_train_samples': len(train_data),
#                 'used_test_samples': len(test_data),
#                 'sampling_enabled': enable_sampling,
#                 'sampling_ratio': f"{train_sample_ratio:.2f}/{test_sample_ratio:.2f}" if enable_sampling else "N/A",
#                 'features': train_data.shape[1],
#                 **metrics
#             }
#             results.append(result)
#             logger.info(f"Results for {item_id}: F1={metrics['f1']:.4f}, AUC={metrics['auc_roc']:.4f}")
#
#             # 可视化结果
#             try:
#                 plot_success = evaluator.plot_results(
#                     test_data, scores, aligned_labels,
#                     metrics['threshold'], f"{dataset_type}_{item_id}"
#                 )
#                 if not plot_success:
#                     logger.warning(f"Plotting failed for {item_id}")
#             except Exception as e:
#                 logger.error(f"Error during plotting for {item_id}: {str(e)}")
#
#             # 清理内存
#             del train_data, test_data, test_labels, scores, aligned_labels
#             gc.collect()
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#
#         except Exception as e:
#             logger.error(f"Error processing item {item_id}: {str(e)}")
#             results.append({
#                 'dataset': dataset_type,
#                 'item': item_id,
#                 'error': str(e),
#                 'f1': np.nan,
#                 'auc_roc': np.nan,
#                 'threshold': np.nan
#             })
#
#     return results
#
# # def run_experiment(logger):
# #     """
# #     运行实验
# #     根据配置处理指定数据集
# #
# #     返回:
# #     results: 结果列表
# #     """
# #     dataset_type = DATA_CONFIG['dataset_type']
# #     dataset_config = DATA_CONFIG.get(dataset_type, {})
# #
# #     logger.info(f"Starting experiment for {dataset_type} dataset")
# #
# #     # 获取数据加载器
# #     try:
# #         loader = get_data_loader(
# #             dataset_type=dataset_type,
# #             root_dir=dataset_config['root_dir'],
# #             config=dataset_config, # 传递完整配置
# #             dataset=dataset_config.get('dataset', 'MSL')  # 仅NASA需要
# #         )
# #     except Exception as e:
# #         logger.error(f"Failed to initialize data loader: {str(e)}")
# #         return []
# #
# #     # 获取所有项目
# #     all_items = loader.get_all_items()
# #
# #     # 确定要处理的项目列表
# #     if dataset_config.get('test_items'):  # 通用处理
# #         items = [i for i in dataset_config['test_items'] if i in all_items]
# #     else:
# #         # 处理所有项目（可能排除某些项目）
# #         exclude_items = dataset_config.get('exclude_items', [])
# #         items = [i for i in all_items if i not in exclude_items]
# #
# #     # 限制最大项目数
# #     max_items = dataset_config.get('max_items')
# #     if max_items is not None and max_items > 0:
# #         items = items[:max_items]
# #
# #     logger.info(f"Found {len(all_items)} items in {dataset_type} dataset")
# #     logger.info(f"Processing {len(items)} items: {', '.join(items[:5])}{'...' if len(items) > 5 else ''}")
# #
# #     # 初始化评估器
# #     evaluator = Evaluator(
# #         save_plots=EXPERIMENT_CONFIG['save_plots'],
# #         plot_dir=EXPERIMENT_CONFIG['plot_dir'],
# #         max_plot_points=EXPERIMENT_CONFIG.get('max_plot_points', 5000)
# #     )
# #
# #     results = []
# #
# #     # 获取采样配置
# #     sampling_config = EXPERIMENT_CONFIG.get('data_sampling', {})
# #     enable_sampling = sampling_config.get('enable', False)
# #     logger.info(f"Data sampling {'ENABLED' if enable_sampling else 'DISABLED'}")
# #
# #     # 处理所有选定项目
# #     for i, item_id in enumerate(items):
# #         logger.info(f"\nProcessing item {i + 1}/{len(items)}: {item_id}")
# #
# #         try:
# #             # 加载数据
# #             train_data, test_data, test_labels = loader.load_item(item_id)
# #
# #             # 检查数据有效性
# #             if len(train_data) == 0 or len(test_data) == 0:
# #                 raise ValueError("Empty data loaded")
# #
# #             # 原始数据大小记录
# #             orig_train_size = len(train_data)
# #             orig_test_size = len(test_data)
# #             logger.info(f"Original data sizes - Train: {orig_train_size}, Test: {orig_test_size}")
# #
# #             # ================= 新增：特征标准化 =================
# #             logger.info("Applying feature scaling...")
# #             from sklearn.preprocessing import StandardScaler
# #
# #             # 注意：仅使用训练数据拟合缩放器
# #             scaler = StandardScaler()
# #             train_data = scaler.fit_transform(train_data)
# #             test_data = scaler.transform(test_data)
# #
# #             # 数据采样 (快速验证模式)
# #             if enable_sampling:
# #                 logger.info(f"Applying data sampling for {item_id}")
# #
# #                 # 训练数据采样
# #                 train_sample_ratio = sampling_config.get('train_sample_ratio', 0.1)
# #                 max_train_points = sampling_config.get('max_train_points', 5000)
# #                 if train_sample_ratio < 1.0:
# #                     n_train = min(int(len(train_data) * train_sample_ratio), max_train_points)
# #                     indices = np.random.choice(len(train_data), n_train, replace=False)
# #                     train_data = train_data[indices]
# #                     logger.info(f"Sampled training data: {len(train_data)}/{orig_train_size} points")
# #                 else:
# #                     logger.info("Using full training data")
# #
# #                 # 测试数据采样
# #                 test_sample_ratio = sampling_config.get('test_sample_ratio', 0.1)
# #                 max_test_points = sampling_config.get('max_test_points', 10000)
# #                 if test_sample_ratio < 1.0:
# #                     n_test = min(int(len(test_data) * test_sample_ratio), max_test_points)
# #                     indices = np.random.choice(len(test_data), n_test, replace=False)
# #                     test_data = test_data[indices]
# #                     test_labels = test_labels[indices]
# #                     logger.info(f"Sampled test data: {len(test_data)}/{orig_test_size} points")
# #                 else:
# #                     logger.info("Using full test data")
# #
# #             # 初始化管道
# #             pipeline = AnomalyDetectionPipeline(
# #                 model_params=MODEL_CONFIG,
# #                 device=EXPERIMENT_CONFIG['device']
# #             )
# #
# #             # 训练和评估
# #             logger.info("Starting training and evaluation...")
# #             scores, aligned_labels = pipeline.fit(train_data, test_data, test_labels,logger)
# #             logger.info("Training and evaluation completed")
# #
# #             # 计算指标
# #             metrics = evaluator.calculate_metrics(scores, aligned_labels)
# #
# #             # 存储结果
# #             result = {
# #                 'dataset': dataset_type,
# #                 'item': item_id,
# #                 'orig_train_samples': orig_train_size,
# #                 'orig_test_samples': orig_test_size,
# #                 'used_train_samples': len(train_data),
# #                 'used_test_samples': len(test_data),
# #                 'sampling_enabled': enable_sampling,
# #                 'sampling_ratio': f"{train_sample_ratio:.2f}/{test_sample_ratio:.2f}" if enable_sampling else "N/A",
# #                 'features': train_data.shape[1],
# #                 **metrics
# #             }
# #             results.append(result)
# #             logger.info(f"Results for {item_id}: F1={metrics['f1']:.4f}, AUC={metrics['auc_roc']:.4f}")
# #
# #             # 可视化结果
# #             try:
# #                 plot_success = evaluator.plot_results(
# #                     test_data, scores, aligned_labels,
# #                     metrics['threshold'], f"{dataset_type}_{item_id}"
# #                 )
# #                 if not plot_success:
# #                     logger.warning(f"Plotting failed for {item_id}")
# #             except Exception as e:
# #                 logger.error(f"Error during plotting for {item_id}: {str(e)}")
# #
# #             # 清理内存
# #             del train_data, test_data, test_labels, scores, aligned_labels
# #             gc.collect()
# #             if torch.cuda.is_available():
# #                 torch.cuda.empty_cache()
# #
# #         except Exception as e:
# #             logger.error(f"Error processing item {item_id}: {str(e)}")
# #             results.append({
# #                 'dataset': dataset_type,
# #                 'item': item_id,
# #                 'error': str(e),
# #                 'f1': np.nan,
# #                 'auc_roc': np.nan,
# #                 'threshold': np.nan
# #             })
# #
# #     return results
#
#
# def main():
#     """主程序"""
#     # 设置日志
#     logger = setup_logger()
#     logger.info("Starting anomaly detection experiment")
#     logger.info(f"Device: {EXPERIMENT_CONFIG['device']}")
#     logger.info(f"Model config: {MODEL_CONFIG}")
#
#     # 确保结果目录存在
#     os.makedirs(os.path.dirname(EXPERIMENT_CONFIG['results_file']), exist_ok=True)
#     os.makedirs(EXPERIMENT_CONFIG['plot_dir'], exist_ok=True)
#
#     start_time = time.time()
#     all_results = []
#
#     # 遍历所有数据集配置
#     for dataset_config in DATA_CONFIG:
#         dataset_name = dataset_config['name']
#         dataset_type = dataset_config['type']
#         dataset_params = dataset_config['params']
#
#         logger.info("\n" + "=" * 50)
#         logger.info(f"Processing {dataset_name} Dataset")
#         logger.info("=" * 50)
#         logger.info(f"Dataset type: {dataset_type}")
#
#         # 应用数据集特定的采样配置
#         sampling_config = EXPERIMENT_CONFIG['data_sampling'].copy()
#         if 'dataset_specific_sampling' in EXPERIMENT_CONFIG:
#             if dataset_name in EXPERIMENT_CONFIG['dataset_specific_sampling']:
#                 # 合并配置
#                 sampling_config.update(
#                     EXPERIMENT_CONFIG['dataset_specific_sampling'][dataset_name]
#                 )
#
#         # 保存原始采样配置
#         original_sampling = EXPERIMENT_CONFIG['data_sampling']
#         # 临时更新采样配置
#         EXPERIMENT_CONFIG['data_sampling'] = sampling_config
#
#         try:
#             # 运行实验
#             results = run_experiment(logger, dataset_type, dataset_params)
#
#             # 为结果添加数据集名称
#             for res in results:
#                 res['dataset_name'] = dataset_name
#                 res['dataset_type'] = dataset_type
#
#             all_results.extend(results)
#
#             # 保存中间结果
#             interim_df = pd.DataFrame(all_results)
#             interim_file = EXPERIMENT_CONFIG['results_file'].replace('.csv', '_interim.csv')
#             interim_df.to_csv(interim_file, index=False)
#             logger.info(f"Interim results saved to {interim_file}")
#
#         except Exception as e:
#             logger.error(f"Error processing dataset {dataset_name}: {str(e)}")
#             all_results.append({
#                 'dataset_name': dataset_name,
#                 'dataset_type': dataset_type,
#                 'error': str(e)
#             })
#         finally:
#             # 恢复原始采样配置
#             EXPERIMENT_CONFIG['data_sampling'] = original_sampling
#             # 清理内存
#             gc.collect()
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#
#     # 保存最终结果
#     results_df = pd.DataFrame(all_results)
#     results_file = EXPERIMENT_CONFIG['results_file']
#     results_df.to_csv(results_file, index=False)
#     logger.info(f"Final results saved to {results_file}")
#
#     # 打印汇总结果
#     logger.info("\n" + "=" * 50)
#     logger.info("Summary of All Results")
#     logger.info("=" * 50)
#
#     # 按数据集分组汇总
#     for dataset_name, group in results_df.groupby('dataset_name'):
#         valid_results = group.dropna(subset=['f1'])
#         failed_count = len(group) - len(valid_results)
#
#         logger.info(f"\n{dataset_name}:")
#         logger.info(f"  Items processed: {len(group)}")
#         logger.info(f"  Successful items: {len(valid_results)}")
#         logger.info(f"  Failed items: {failed_count}")
#
#         if not valid_results.empty:
#             avg_f1 = valid_results['f1'].mean()
#             avg_auc = valid_results['auc_roc'].mean()
#             logger.info(f"  Average F1: {avg_f1:.4f}")
#             logger.info(f"  Average AUC-ROC: {avg_auc:.4f}")
#
#             # F1分数分布
#             logger.info(f"  F1 distribution:")
#             logger.info(f"    Min: {valid_results['f1'].min():.4f}")
#             logger.info(f"    25%: {valid_results['f1'].quantile(0.25):.4f}")
#             logger.info(f"    Median: {valid_results['f1'].median():.4f}")
#             logger.info(f"    75%: {valid_results['f1'].quantile(0.75):.4f}")
#             logger.info(f"    Max: {valid_results['f1'].max():.4f}")
#
#     # 显示处理失败的通道
#     failed_results = results_df[results_df['f1'].isna()]
#     if not failed_results.empty:
#         logger.info("\nFailed items summary:")
#         for _, row in failed_results.iterrows():
#             logger.info(f"- {row['dataset_name']}: {row['item']} - {row.get('error', 'Unknown error')}")
#
#     # 显示总耗时
#     elapsed = time.time() - start_time
#     logger.info(f"\nTotal processing time: {elapsed / 60:.2f} minutes")
#
#
# # def main():
# #     """主程序"""
# #     # 设置日志
# #     logger = setup_logger()
# #     logger.info("Starting anomaly detection experiment")
# #
# #     # 确保结果目录存在
# #     os.makedirs(os.path.dirname(EXPERIMENT_CONFIG['results_file']), exist_ok=True)
# #     os.makedirs(EXPERIMENT_CONFIG['plot_dir'], exist_ok=True)
# #
# #     start_time = time.time()
# #     all_results = []
# #
# #     # 定义要运行的所有数据集配置
# #     dataset_configs = [
# #         # NASA 数据集 (MSL 和 SMAP)
# #         {'type': 'nasa', 'dataset': 'MSL', 'name': 'NASA_MSL'},
# #         {'type': 'nasa', 'dataset': 'SMAP', 'name': 'NASA_SMAP'},
# #
# #         # SMD 数据集
# #         {'type': 'smd', 'name': 'SMD'},
# #
# #         # SWAT 数据集
# #         {'type': 'swat', 'name': 'SWAT'},
# #
# #         # WADI 数据集
# #         {'type': 'wadi', 'name': 'WADI'}
# #     ]
# #
# #
# #
# #     # 运行实验
# #     try:
# #         for config in dataset_configs:
# #             dataset_type = config['type']
# #             dataset_name = config['name']
# #
# #             logger.info("\n" + "=" * 50)
# #             logger.info(f"Processing {dataset_name} Dataset")
# #             logger.info("=" * 50)
# #
# #             # 获取数据集特定配置
# #             if dataset_type in DATA_CONFIG:
# #                 dataset_config = DATA_CONFIG[dataset_type].copy()
# #             else:
# #                 logger.warning(f"No specific config found for {dataset_type}, using default")
# #                 dataset_config = {}
# #
# #             # NASA 需要额外参数
# #             if dataset_type == 'nasa':
# #                 dataset_config['dataset'] = config['dataset']
# #
# #             # 设置当前数据集类型
# #             DATA_CONFIG['dataset_type'] = dataset_type
# #
# #             # 运行实验
# #             results = run_experiment(logger)
# #
# #             # 为结果添加数据集名称
# #             for res in results:
# #                 res['dataset_name'] = dataset_name
# #                 res['dataset_type'] = dataset_type
# #                 if dataset_type == 'nasa':
# #                     res['dataset_subtype'] = config['dataset']
# #
# #             all_results.extend(results)
# #
# #             # 保存中间结果
# #             interim_df = pd.DataFrame(all_results)
# #             interim_file = EXPERIMENT_CONFIG['results_file'].replace('.csv', '_interim.csv')
# #             interim_df.to_csv(interim_file, index=False)
# #             logger.info(f"Interim results saved to {interim_file}")
# #
# #             # 保存最终结果
# #         results_df = pd.DataFrame(all_results)
# #         results_file = EXPERIMENT_CONFIG['results_file']
# #         results_df.to_csv(results_file, index=False)
# #         logger.info(f"Final results saved to {results_file}")
# #
# #         # 打印汇总结果
# #         logger.info("\n" + "=" * 50)
# #         logger.info("Summary of All Results")
# #         logger.info("=" * 50)
# #
# #         # 按数据集分组汇总
# #         for dataset_name, group in results_df.groupby('dataset_name'):
# #             valid_results = group.dropna(subset=['f1'])
# #             if not valid_results.empty:
# #                 avg_f1 = valid_results['f1'].mean()
# #                 avg_auc = valid_results['auc_roc'].mean()
# #                 logger.info(f"\n{dataset_name}:")
# #                 logger.info(f"  Items processed: {len(valid_results)}")
# #                 logger.info(f"  Average F1: {avg_f1:.4f}")
# #                 logger.info(f"  Average AUC-ROC: {avg_auc:.4f}")
# #
# #         # 显示处理失败的通道
# #         failed_results = results_df[results_df['f1'].isna()]
# #         if not failed_results.empty:
# #             logger.info("\nFailed items:")
# #             for _, row in failed_results.iterrows():
# #                 logger.info(f"- {row['dataset_name']}: {row['item']} - {row.get('error', 'Unknown error')}")
# #
# #         # 显示总耗时
# #         elapsed = time.time() - start_time
# #         logger.info(f"\nTotal processing time: {elapsed / 60:.2f} minutes")
# #
# #     except Exception as e:
# #         logger.exception("Unhandled exception in main process")
# #
# #     #     logger.info("\n" + "=" * 50)
# #     #     logger.info(f"Processing {DATA_CONFIG['dataset_type'].upper()} Dataset")
# #     #     logger.info("=" * 50)
# #     #
# #     #     results = run_experiment(logger)
# #     #
# #     #     # 保存结果
# #     #     results_df = pd.DataFrame(results)
# #     #     results_file = EXPERIMENT_CONFIG['results_file']
# #     #     results_df.to_csv(results_file, index=False)
# #     #     logger.info(f"Results saved to {results_file}")
# #     #
# #     #     # 打印汇总结果
# #     #     logger.info("\n" + "=" * 50)
# #     #     logger.info("Summary of Results")
# #     #     logger.info("=" * 50)
# #     #     logger.info(f"\n{results_df.to_string()}")
# #     #
# #     #     # 计算平均指标（排除失败项）
# #     #     valid_results = results_df.dropna(subset=['f1'])
# #     #     if not valid_results.empty:
# #     #         avg_f1 = valid_results['f1'].mean()
# #     #         avg_auc = valid_results['auc_roc'].mean()
# #     #         logger.info(f"\nAverage F1 Score (across {len(valid_results)} items): {avg_f1:.4f}")
# #     #         logger.info(f"Average AUC-ROC (across {len(valid_results)} items): {avg_auc:.4f}")
# #     #
# #     #     # 显示处理失败的通道
# #     #     failed_results = results_df[results_df['f1'].isna()]
# #     #     if not failed_results.empty:
# #     #         logger.info(f"\nFailed to process {len(failed_results)} items:")
# #     #         for _, row in failed_results.iterrows():
# #     #             logger.info(f"- {row['item']}: {row.get('error', 'Unknown error')}")
# #     #
# #     #     # 显示总耗时
# #     #     elapsed = time.time() - start_time
# #     #     logger.info(f"\nTotal processing time: {elapsed / 60:.2f} minutes")
# #     # except Exception as e:
# #     #     logger.exception("Unhandled exception in main process")
#
#
# if __name__ == "__main__":
#     main()