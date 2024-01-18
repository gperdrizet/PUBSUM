import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import notebook_helper_functions.data_analysis_functions as data_funcs
from matplotlib.ticker import FormatStrFormatter
from typing import List, Union

pd.options.mode.chained_assignment = None

def baseline_execution_plot(datafile):
    
    data = pd.read_csv(datafile)

    titles = ['Total', 'Summarization', 'SQL insert', 'SQL load']
    data_types = ['mean_total_time', 'mean_summarization_time', 'mean_insert_time', 'mean_loading_time']
    xlabels = ['seconds', 'seconds', 'milliseconds', 'milliseconds']
    xaxis_scales = [1, 1, 10**3, 10**3]

    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True, tight_layout=True)

    fig.suptitle('Execution time distributions')
    fig.supxlabel('Execution time')

    axs_num = 0

    for title, data_type, xlabel, xaxis_scale in zip(titles, data_types, xlabels, xaxis_scales):

        axs[axs_num].set_title(title)
        axs[axs_num].set_xlabel(xlabel)
        axs[axs_num].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        axs[axs_num].hist(
            data[data_type] * xaxis_scale,
            histtype='stepfilled', 
            cumulative=0,
            color='black'
        )

        axs_num += 1

    axs[0].set_ylabel('Replicates (n)')

    return data, plt


def device_map_plot(datafile):

    data = pd.read_csv(datafile)
    data['summarization rate (abstracts/min.)'] = data['summarization rate (abstracts/sec.)'] * 60

    wide_data = data.pivot(index='replicate', columns='device map strategy', values='summarization rate (abstracts/min.)')

    fig, axs = plt.subplots(1, 1, figsize=(9, 3), tight_layout=True)

    axs.set_title('Device map strategy benchmark')
    axs.set_xlabel('Huggingface device map')
    axs.set_ylabel('Summarization rate\n(abstracts/minute)')
    axs.boxplot(wide_data, labels=wide_data.columns)

    return data, plt


def parallel_summarization_plot(datafile):

    data = pd.read_csv(datafile)

    OOM_columns = [
        'summarization time (sec.)',
        'summarization rate (abstracts/sec.)',
        'max memory allocated (bytes)',
        'model memory footprint (bytes)'
    ]

    OOM = data[data['summarization time (sec.)'] == 'OOM']
    OOM = OOM[['device', 'workers']]
    OOM_conditions = OOM.to_numpy().tolist()

    for OOM_column in OOM_columns:
        for condition in OOM_conditions:

            data[OOM_column].loc[
                (data['device'] == condition[0]) &
                (data['workers'] == condition[1])
            ] = np.nan

    data.dropna(inplace=True)

    data['summarization time (sec.)'] = data['summarization time (sec.)'].astype(float)
    data['summarization rate (abstracts/sec.)'] = data['summarization rate (abstracts/sec.)'].astype(float)
    data['max memory allocated (bytes)'] = data['max memory allocated (bytes)'].astype(int)
    data['model memory footprint (bytes)'] = data['model memory footprint (bytes)'].astype(int)

    data['summarization rate (abstracts/min.)'] = data['summarization rate (abstracts/sec.)'] * 60
    data['max memory allocated (GB)'] = data['max memory allocated (bytes)'] / 10 ** 9
    data['model memory footprint (GB)'] = data['model memory footprint (bytes)'] / 10 ** 9

    devices = ['GPU', 'CPU: 1 thread per worker', 'CPU: 2 threads per worker', 'CPU: 4 threads per worker']

    fig, axs = plt.subplots(1, 2, figsize=(11, 5), tight_layout=True)

    axs[0].set_title('Summarization rate')
    axs[0].set_xlabel('Concurrent worker processes')
    axs[0].set_ylabel('Summarization rate\n(abstracts/minute)')
    axs[0].set_yscale('log', base=2)
    axs[0].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    for device in devices:

        plot_data = data[data['device'] == device]

        mean = plot_data.groupby(['device', 'workers']).mean()
        mean.reset_index(inplace=True)
        
        std = plot_data.groupby(['device', 'workers']).std()
        std.reset_index(inplace=True)

        axs[0].errorbar(
            mean['workers'], 
            mean['summarization rate (abstracts/min.)'], 
            yerr=std['summarization rate (abstracts/min.)'],
            capsize=5,
            label=device,
            linestyle='dotted'
        )

    axs[1].set_title('Memory use')
    axs[1].set_xlabel('Concurrent worker processes')
    axs[1].set_ylabel('Total max memory allocated\n(GB)')
    axs[1].set_yscale('log', base=2)
    axs[1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    for device in devices:

        plot_data = data[data['device'] == device]

        mean = plot_data.groupby(['device', 'workers']).mean()
        mean.reset_index(inplace=True)
        
        std = plot_data.groupby(['device', 'workers']).std()
        std.reset_index(inplace=True)

        axs[1].errorbar(
            mean['workers'], 
            mean['max memory allocated (GB)'], 
            yerr=std['max memory allocated (GB)'],
            capsize=5,
            label=device,
            linestyle='dotted'
        )

    plt.legend(loc='best')

    return data, plt


def model_quantization_plot(datafile):

    data = pd.read_csv(datafile)
    data['summarization rate (abstracts/min.)'] = data['summarization rate (abstracts/sec.)'] * 60
    data['model GPU memory footprint (GB)'] = data['model GPU memory footprint (bytes)'] / 10 ** 9

    fig, axs = plt.subplots(2, 1, figsize=(9, 8), tight_layout=True)

    mean_memory_footprint_data = data[['quantization strategy', 'model GPU memory footprint (GB)']].groupby('quantization strategy').mean()

    axs[0].set_title('Model quantization benchmark: model memory footprint')
    axs[0].set_xlabel('Quantization strategy')
    axs[0].set_ylabel('GPU memory (GB)')
    axs[0].tick_params(axis='x', labelrotation=45)
    axs[0].bar(
        x=mean_memory_footprint_data.index, 
        height=mean_memory_footprint_data['model GPU memory footprint (GB)'],
        color='black',
        fill=False
    )

    rate_data = data[['replicate', 'quantization strategy', 'summarization rate (abstracts/min.)']]
    rate_data = data.pivot(index='replicate', columns='quantization strategy', values='summarization rate (abstracts/min.)')

    axs[1].set_title('Model quantization benchmark: summarization rate')
    axs[1].set_xlabel('Quantization strategy')
    axs[1].set_ylabel('Summarization rate\n(abstracts/minute)')
    axs[1].tick_params(axis='x', labelrotation=45)
    axs[1].boxplot(rate_data, labels=rate_data.columns)

    return data, plt


def batch_summarization_plot(
    datafile: str,
    unique_condition_columns: List[str],
    quantization_method: str,
    oom_columns: List[str], 
    str_columns: List[str], 
    int_columns: List[str], 
    float_columns: List[str],
    oom_replacement_val: Union[str, int, float]
):

    # Read data
    data = pd.read_csv(datafile)

    # Clean out-of-memory errors and replace with user defined value
    data = data_funcs.clean_out_of_memory_errors(
        data=data, 
        unique_condition_columns=unique_condition_columns,
        oom_columns=oom_columns, 
        str_columns=str_columns, 
        int_columns=int_columns, 
        float_columns=float_columns,
        oom_replacement_val=oom_replacement_val
    )

    # Clean up any leftover NANs
    data.dropna(axis=0, inplace=True)

    # Do some unit conversion
    data['summarization rate (abstracts/min.)'] = data['summarization rate (abstracts/sec.)'] * 60
    data['max memory allocated (GB)'] = data['max memory allocated (bytes)'] / 10 ** 9

    # Get min and max rate values dataset wide so we can set common axis limits
    #max_memory = math.ceil(max(data['max memory allocated (GB)']))
    max_rate = math.ceil(max(data['summarization rate (abstracts/min.)']))
    min_rate = math.floor(min(data['summarization rate (abstracts/min.)']))

    # Set figure layout - first column is un-quantized data, second column is quantized data
    fig, axs = plt.subplots(2, 2, figsize=(9, 9), tight_layout=True)

    # Split off un-quantized data for first column
    unquantized_data = data[data['quantization'] == 'none']

    # Get mean and standard deviation of memory use data for plotted values
    mean_max_memory_data = unquantized_data[['batch size', 'max memory allocated (GB)']].groupby('batch size').mean()
    std_max_memory_data = unquantized_data[['batch size', 'max memory allocated (GB)']].groupby('batch size').std()

    axs[0, 0].set_title('Max memory allocated: un-quantized model')
    axs[0, 0].set_xlabel('Batch size')
    axs[0, 0].set_ylabel('GPU memory (GB)')
    axs[0, 0].set_xlim([-0.95, 8])
    axs[0, 0].set_ylim([0, 12]) # Note: this is the total memory of one K80 chip - can we do better?
    # axs[0, 0].hlines(y=11.4, xmin=-0.95, xmax=8, linewidth=1, color='red')
    # axs[0, 0].hlines(y=3132600320 / 10 ** 9, xmin=-0.95, xmax=8, linewidth=1, color='y')
    #axs[0, 0].annotate('Model\nFootprint', xy=(-0.9, 4), color='red')
    #axs[0, 0].annotate('OOM', xy=(5.1, 1), color='black')
    axs[0, 0].bar(
        x=list(range(len(mean_max_memory_data['max memory allocated (GB)']))), 
        height=mean_max_memory_data['max memory allocated (GB)'],
        yerr=std_max_memory_data['max memory allocated (GB)'] * 3,
        capsize=5,
        tick_label=mean_max_memory_data.index,
        color='black', 
        fill=False
    )

    # Get and re-format rate data from unquantized replicates 
    rate_data = unquantized_data[['replicate', 'batch size', 'summarization rate (abstracts/min.)']]
    rate_data = rate_data.pivot(index='replicate', columns='batch size', values='summarization rate (abstracts/min.)')

    axs[1, 0].set_title('Summarization rate: un-quantized model')
    axs[1, 0].set_xlabel('Batch size')
    axs[1, 0].set_ylabel('Summarization rate\n(abstracts/minute)')
    #axs[1, 0].annotate('OOM', xy=(6.1, 23), color='black')
    #axs[1, 0].set_xlim([])
    axs[1, 0].set_ylim([min_rate, max_rate])
    axs[1, 0].boxplot(
        rate_data, 
        labels=rate_data.columns, 
        medianprops=dict(color='red')
    )

    # Repeat above for quantized data - can we refactor this into two plot calls instead of 4 total, individual calls
    quantized_data = data[data['quantization'] == quantization_method]

    mean_max_memory_data = quantized_data[['batch size', 'max memory allocated (GB)']].groupby('batch size').mean()
    std_max_memory_data = quantized_data[['batch size', 'max memory allocated (GB)']].groupby('batch size').std()

    axs[0, 1].set_title('Max memory allocated: quantized model')
    axs[0, 1].set_xlabel('Batch size')
    axs[0, 1].set_ylabel('GPU memory (GB)')
    axs[0, 1].set_xlim([-0.95, 8])
    axs[0, 1].set_ylim([0, 12])
    # axs[0, 1].hlines(y=11.4, xmin=-0.95, xmax=8, linewidth=1, color='red')
    # axs[0, 1].hlines(y=974903296 / 10 ** 9, xmin=-0.95, xmax=8, linewidth=1, color='y')
    #axs[0, 1].annotate('Model\nFootprint', xy=(-0.9, 1.5), color='red')
    axs[0, 1].bar(
        x=list(range(len(mean_max_memory_data['max memory allocated (GB)']))), 
        height=mean_max_memory_data['max memory allocated (GB)'],
        yerr=std_max_memory_data['max memory allocated (GB)'] * 3,
        capsize=5,
        tick_label=mean_max_memory_data.index,
        color='black', 
        fill=False
    )

    rate_data = quantized_data[['replicate', 'batch size', 'summarization rate (abstracts/min.)']]
    rate_data = quantized_data.pivot(index='replicate', columns='batch size', values='summarization rate (abstracts/min.)')

    axs[1, 1].set_title('Summarization rate: quantized model')
    axs[1, 1].set_xlabel('Batch size')
    axs[1, 1].set_ylabel('Summarization rate\n(abstracts/minute)')
    axs[1, 1].set_ylim([min_rate, max_rate])
    axs[1, 1].boxplot(
        rate_data, 
        labels=rate_data.columns, 
        medianprops=dict(color='red')
    )

    return data, plt


def parallel_batched_summarization_plot(datafile):

    data = pd.read_csv(datafile)

    OOM_columns = [
        'summarization time (sec.)',
        'summarization rate (abstracts/sec.)'
    ]

    OOM = data[data['summarization time (sec.)'] == 'OOM']
    OOM = OOM[['batch size', 'workers', 'quantization']]
    OOM_conditions = OOM.to_numpy().tolist()

    for OOM_column in OOM_columns:
        for condition in OOM_conditions:

            data[OOM_column].loc[
                (data['batch size'] == condition[0]) &
                (data['workers'] == condition[1]) &
                (data['quantization'] == condition[2])
            ] = np.nan

    data.dropna(inplace=True)

    data['summarization time (sec.)'] = data['summarization time (sec.)'].astype(float)
    data['summarization rate (abstracts/sec.)'] = data['summarization rate (abstracts/sec.)'].astype(float)
    data['max memory allocated (bytes)'] = data['max memory allocated (bytes)'].astype(float)
    data['model memory footprint (bytes)'] = data['model memory footprint (bytes)'].astype(float)

    data['summarization rate (abstracts/min.)'] = data['summarization rate (abstracts/sec.)'] * 60
    data['max memory allocated (GB)'] = data['max memory allocated (bytes)'] / 1024**3
    data['model memory footprint (GB)'] = data['model memory footprint (bytes)'] / 1024**3
    data['jobs per GPU'] = data['workers'] // 4

    max_batch_size = max(data['batch size'])
    min_batch_size = min(data['batch size'])
    max_summarization_rate = max(data['summarization rate (abstracts/min.)'])
    min_summarization_rate = min(data['summarization rate (abstracts/min.)'])
    max_memory = max(data['max memory allocated (GB)'])
    min_memory = min(data['max memory allocated (GB)'])
    axis_pad = 0.1

    quantization_types = data['quantization'].unique()
    worker_nums = data['jobs per GPU'].unique()

    fig, axs = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)

    # Summarization rate plots
    axs_count = 0

    for quantization in quantization_types:
        quantization_type_data = data[data['quantization'] == quantization].copy()
        quantization_type_data.drop('quantization', axis=1, inplace=True)

        for workers in worker_nums:

            plot_data = quantization_type_data[quantization_type_data['jobs per GPU'] == workers]

            mean = plot_data.groupby(['batch size']).mean()
            mean.reset_index(inplace=True)
            
            std = plot_data.groupby(['batch size']).std()
            std.reset_index(inplace=True)

            axs[0, axs_count].set_title(f'Model quantization: {quantization}')
            axs[0, axs_count].set_xlabel('Batch size')
            axs[0, axs_count].set_ylabel('Summarization rate (abstracts/min.)')

            axs[0, axs_count].set_xlim([
                (min_batch_size - (min_batch_size * axis_pad)), 
                (max_batch_size + (max_batch_size * axis_pad))
            ])
            
            axs[0, axs_count].set_ylim([(
                min_summarization_rate - (min_summarization_rate * axis_pad)), 
                (max_summarization_rate + (max_summarization_rate * axis_pad))
            ])
            
            axs[0, axs_count].set_xscale('log', base=2)
            #axs[0, axs_count].set_yscale('log', base=2)
            #axs[0, axs_count].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

            axs[0, axs_count].errorbar(
                mean['batch size'], 
                mean['summarization rate (abstracts/min.)'], 
                yerr=std['summarization rate (abstracts/min.)'],
                capsize=2.5,
                label=workers,
                linestyle='dotted'
            )

        axs_count += 1

    # Memory use plots
    axs_count = 0

    for quantization in quantization_types:
        quantization_type_data = data[data['quantization'] == quantization].copy()
        quantization_type_data.drop('quantization', axis=1, inplace=True)

        for workers in worker_nums:

            plot_data = quantization_type_data[quantization_type_data['jobs per GPU'] == workers]

            mean = plot_data.groupby(['batch size']).mean()
            mean.reset_index(inplace=True)
            
            std = plot_data.groupby(['batch size']).std()
            std.reset_index(inplace=True)

            axs[1, axs_count].set_title(f'Model quantization: {quantization}')
            axs[1, axs_count].set_xlabel('Batch size')
            axs[1, axs_count].set_ylabel('Max allocated memory (GB)')

            axs[1, axs_count].hlines(
                y=(11.4 * 4), 
                xmin=(min_batch_size - (min_batch_size * axis_pad)), 
                xmax=(max_batch_size + (max_batch_size * axis_pad)), 
                linewidth=0.5,
                color='red'
            )

            axs[1, axs_count].set_xlim([
                (min_batch_size - (min_batch_size * axis_pad)), 
                (max_batch_size + (max_batch_size * axis_pad))
            ])
            
            axs[1, axs_count].set_ylim([
                (min_memory - (min_memory * axis_pad)), 
                50 #(max_memory + (max_memory * axis_pad))
            ])
            
            axs[1, axs_count].set_xscale('log', base=2)
            #axs[1, axs_count].set_yscale('log', base=2)
            #axs[1, axs_count].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

            axs[1, axs_count].errorbar(
                mean['batch size'], 
                mean['max memory allocated (GB)'], 
                yerr=std['max memory allocated (GB)'],
                capsize=2.5,
                label=workers,
                linestyle='dotted'
            )

        axs_count += 1

    plt.legend(loc='upper left', title='Workers per GPU')

    return data, plt


def sql_insert_plot(datafile):

    data = pd.read_csv(datafile)
    data['insert rate (abstracts/millisecond)'] = data['insert rate (abstracts/sec.)'] / 1000
    data['thousand abstracts'] = data['abstracts'] / 1000

    insert_strategies = ['execute_many', 'execute_batch', 'execute_values', 'mogrify', 'stringIO']

    fig, axs = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)

    axs.set_title('SQL insert benchmark')
    axs.set_xlabel('Thousand abstracts inserted')
    axs.set_ylabel('Insertion rate\n(abstracts/millisecond)')
    #axs.set_yscale('log', base=2)
    #axs.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    #axs.tick_params(axis='x', labelrotation=45)

    for insert_strategy in insert_strategies:

        plot_data = data[data['insert strategy'] == insert_strategy]

        mean = plot_data.groupby(['insert strategy', 'thousand abstracts']).mean()
        mean.reset_index(inplace=True)
        
        std = plot_data.groupby(['insert strategy', 'thousand abstracts']).std()
        std.reset_index(inplace=True)

        axs.errorbar(
            mean['thousand abstracts'], 
            mean['insert rate (abstracts/millisecond)'], 
            yerr=std['insert rate (abstracts/millisecond)'],
            capsize=5,
            label=insert_strategy,
            linestyle='dotted'
        )

    plt.legend(loc='center right')

    return data, plt