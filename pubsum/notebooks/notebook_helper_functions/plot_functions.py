import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import notebook_helper_functions.data_analysis_functions as data_funcs
from matplotlib.ticker import FormatStrFormatter
from typing import List, Union
from textwrap import wrap

pd.options.mode.chained_assignment = None

def baseline_execution_plot(data_file: str) -> pd.DataFrame:
    
    # Load data
    data = pd.read_csv(data_file)

    # Get means
    mean_replicate_time = (data['replicate_time'].mean()).round(1)
    mean_summarization_time = (data['summarization_time'].mean()).round(1)
    mean_insert_time = (data['insert_time'].mean()).round(3)
    mean_load_time = (data['loading_time'].mean()).round(2)

    # Lists of plot parameters to loop on
    titles = ['Total', 'Summarization', 'SQL load', 'SQL insert']
    data_types = ['replicate_time', 'summarization_time', 'loading_time', 'insert_time']
    means = [mean_replicate_time, mean_summarization_time, mean_load_time, mean_insert_time]
    mean_text_x_offsets = [0, 0, 0.1, 0]
    xlabels = ['seconds', 'seconds', 'seconds', 'seconds']
    xaxis_scales = [1, 1, 1, 1]
    xaxis_decimal_places = [0, 0, 1, 2]

    # Set general font size
    plt.rcParams['font.size'] = '16'

    # Set up plots
    fig, axs = plt.subplots(1, 4, figsize=(14, 4), sharey=True, tight_layout=True)

    # Add text to the figure
    fig.suptitle('Unoptimized run time')
    fig.supxlabel('seconds')

    # Loop on the lists of plot vars, counting the axis number
    axs_num = 0

    for title, data_type, mean, mean_text_x_offset, xlabel, xaxis_scale, xaxis_decimals in zip(titles, data_types, means, mean_text_x_offsets, xlabels, xaxis_scales, xaxis_decimal_places):

        # Set tick font size
        for label in (axs[axs_num].get_xticklabels() + axs[axs_num].get_yticklabels()):
            label.set_fontsize(14)

        # Set plot titles and labels
        axs[axs_num].set_title(title)
        axs[axs_num].xaxis.set_major_formatter(FormatStrFormatter(f'%.{xaxis_decimals}f'))

        # Make the plot
        axs[axs_num].hist(
            data[data_type] * xaxis_scale,
            histtype='stepfilled', 
            cumulative=0,
            color='black'
        )

        # Set title inside plot area
        axs[axs_num].annotate(
            f'mean = {mean} s', 
            xy=(0 + mean_text_x_offset, 1),
            xytext=(12, -12),
            va='top',
            xycoords='axes fraction',
            textcoords='offset points',
            fontsize=14
        )

        # Increment to the next axis
        axs_num += 1

    # Set x label on first plot, common for all of the plots
    axs[0].set_ylabel('Abstracts (n)', fontsize=16)

    # Draw the plot
    plt.show()

    # return the data, incase we want to do something else with it
    return data

def baseline_execution_pie(data_file: str) -> pd.DataFrame:

    # Load data
    data = pd.read_csv(data_file)

    # Get means
    mean_summarization_time = data['summarization_time'].mean().round(1)
    mean_insert_time = data['insert_time'].mean().round(3)
    mean_load_time = data['loading_time'].mean().round(2)

    fig, axs = plt.subplots(1,1, figsize=(4, 4))

    # Set general font size
    plt.rcParams['font.size'] = '14'

    axs.set_title('Mean run time per abstract')

    # Set tick font size
    for label in (axs.get_xticklabels() + axs.get_yticklabels()):
        label.set_fontsize(12)

    axs.pie(
        [mean_summarization_time, mean_insert_time, mean_load_time],
        labels=[
            f'Summarization', 
            f'SQL insert', 
            f'SQL load'
        ],
        colors=['white', 'black', 'lightgray'],
        wedgeprops={'linewidth': 0.5, 'edgecolor': 'black'}
    )

    plt.show

def device_map_plot(
    data_file: str, 
    show_table: bool ) -> pd.DataFrame:

    data = pd.read_csv(data_file)

    data['summarization rate (abstracts/min.)'] = data['summarization rate (abstracts/sec.)'] * 60
    wide_data = data.pivot(index='replicate', columns='device map strategy', values='summarization rate (abstracts/min.)')

    # Set general font size
    plt.rcParams['font.size'] = '16'

    fig, axs = plt.subplots(1, 1, figsize=(9, 4), tight_layout=True)

    # Set tick font size
    for label in (axs.get_xticklabels() + axs.get_yticklabels()):
        label.set_fontsize(14)

    axs.set_title('Device map strategy benchmark')
    axs.set_xlabel('Huggingface device map')
    axs.set_ylabel('Summarization rate\n(abstracts/minute)')
    axs.boxplot(wide_data, labels=wide_data.columns)

    plt.show()

    if show_table == True:
        table = data.groupby(['device map strategy'])['summarization rate (abstracts/min.)'].mean().round(2)
        table = pd.DataFrame({'device map strategy':table.index, 'mean summarization rate (abstracts/min.)':table.values})
        print(table)

    return data

def parallel_summarization_plot(
    data_file: str,
    show_table: bool,
    plot_vars: List[str],
    plot_devices: List[str],
    legend_entries: List[str],
    unique_condition_columns: List[str],
    oom_columns: List[str], 
    str_columns: List[str], 
    int_columns: List[str], 
    float_columns: List[str],
    oom_replacement_val: Union[str, int, float]) -> pd.DataFrame:

    # Load data
    data = pd.read_csv(data_file)

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

    # Add some new columns with different units
    data['summarization rate (abstracts/min.)'] = data['summarization rate (abstracts/sec.)'] * 60
    data['max memory allocated (GB)'] = data['max memory allocated (bytes)'] / 10 ** 9
    data['model memory footprint (GB)'] = data['model memory footprint (bytes)'] / 10 ** 9

    # Clean up any leftover NANs
    data.dropna(axis=0, inplace=True)

    # Dict to hold titles etc for possible plots we might need to make for this dataset
    plot_text = {
        'summarization rate (abstracts/min.)': {'title': 'Summarization rate', 'ylabel': 'abstracts/minute'},
        'max memory allocated (GB)': {'title': 'Max memory allocated', 'ylabel': 'gigabytes'},
        'model memory footprint (GB)': {'title': 'Model memory footprint', 'ylabel': 'gigabytes'}
    }

    # Set general font size
    plt.rcParams['font.size'] = '16'

    # Figure setup - first determine how many cols we need
    fig_cols = len(plot_vars)

    # Then, derive the width of the fig from the number of cols
    fig_width = 4.5 * fig_cols

    # Finally, make the figure and axes array
    fig, axs = plt.subplots(1, fig_cols, figsize=(fig_width, 4.5), tight_layout=True)

    # Fix the edge case where we only are drawing one plot - in that situation plt returns a single
    # axis object instead of a list with one member causing axis indexing in our plot loop to fail.
    if fig_cols == 1:
        axs = [axs]

    # Loop on variables passed to plot and make each plot
    ax_count = 0
    for plot_var in plot_vars:

        # Set title and axs labels
        axs[ax_count].set_title(plot_text[plot_var]['title'])
        axs[ax_count].set_ylabel(plot_text[plot_var]['ylabel'])
        axs[ax_count].set_xlabel('Concurrent worker processes')
        axs[ax_count].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        for label in (axs[ax_count].get_xticklabels() + axs[ax_count].get_yticklabels()):
            label.set_fontsize(14)

        # Loop on devices for plot and plot each one
        for device in plot_devices:

            # Pull data for that device
            plot_data = data[data['device'] == device]

            # Get means
            mean = plot_data.groupby(['device', 'workers']).mean()
            mean.reset_index(inplace=True)
            
            # Get standard deviations
            std = plot_data.groupby(['device', 'workers']).std()
            std.reset_index(inplace=True)

            # Decide if we need error bars or just scatter
            if sum(std[plot_var]) > 0:

                # Plot the data
                axs[ax_count].errorbar(
                    mean['workers'], 
                    mean[plot_var], 
                    yerr=std[plot_var],
                    capsize=5,
                    label=device,
                    linestyle='dotted'
                )

            else:

                # Plot the data
                axs[ax_count].plot(
                    mean['workers'], 
                    mean[plot_var], 
                    #yerr=std[plot_var],
                    #capsize=5,
                    label=device,
                    linestyle='solid'
                )
        
        # Add legend
        axs[ax_count].legend(legend_entries, loc='lower right', prop={'size': 9})

        # Move to the next plot
        ax_count += 1

    # Draw the plot
    plt.show()

    # Make summary table(s)
    if show_table == True:
        for plot_var in plot_vars:
            print(f'{plot_text[plot_var]["title"]} ({plot_text[plot_var]["ylabel"]})')
            table = data[data['device'].isin(plot_devices)]
            table = table.groupby(['device', 'workers'])[plot_var].mean().round(2)
            table = table.unstack()
            print(f'{table}\n')

    return data

def model_quantization_plot(
    data_file: str,
    show_table: bool) -> pd.DataFrame:

    # Load data
    data = pd.read_csv(data_file)

    # Add some derived columns with different units
    data['summarization rate (abstracts/min.)'] = data['summarization rate (abstracts/sec.)'] * 60
    data['model GPU memory footprint (GB)'] = data['model GPU memory footprint (bytes)'] / 10 ** 9
    data['max memory allocated (GB)'] = data['max memory allocated (bytes)'] / 10 ** 9

    # Set general font size
    plt.rcParams['font.size'] = '16'

    # Set up the figure and axes array
    fig, axs = plt.subplots(2, 1, figsize=(14, 7), sharex=True, tight_layout=True, gridspec_kw = {'wspace':0, 'hspace':0})
    #plt.subplots_adjust(hspace=0)

    # Format the data for boxplot
    plot_data = data[['replicate', 'quantization strategy', 'summarization rate (abstracts/min.)']]
    plot_data = data.pivot(index='replicate', columns='quantization strategy', values='summarization rate (abstracts/min.)')

    # Make summarization rate boxplot
    axs[0].boxplot(
        plot_data,
        positions=range(len(plot_data.columns)),
        labels=plot_data.columns,
        widths=0.8 # Set box widths to match default for bar plot
    )

    # Set title inside plot area
    axs[0].annotate(
        'Summarization rate', 
        xy=(0, 1),
        xytext=(12, -12),
        va='top',
        xycoords='axes fraction',
        textcoords='offset points',
        fontsize=16
    )

    # Other labels
    axs[0].set_ylabel('Abstracts/min.')
    axs[0].tick_params(axis='x', labelrotation=45)

    # Get mean memory for plotting
    max_memory_footprint_data = data[['quantization strategy', 'model GPU memory footprint (GB)']].groupby('quantization strategy').max()

    # Make the bar plot for memory use
    axs[1].bar(
        x=plot_data.columns, 
        height=max_memory_footprint_data['model GPU memory footprint (GB)'],
        color='black',
        fill=True
    )

    # Set title inside plot area
    axs[1].annotate(
        'Model memory footprint', 
        xy=(0, 1),
        xytext=(12, -12),
        va='top',
        xycoords='axes fraction',
        textcoords='offset points',
        fontsize=16
    )

    # Format x axis labeling
    xtick_labels = list(plot_data.columns)
    labels = [ '\n'.join(wrap(label, 10)) for label in xtick_labels]

    axs[1].set_xticklabels(labels)
    axs[1].tick_params(axis='x', labelrotation=45)
    axs[1].set_ylabel('GPU memory (GB)')

    fig.text(0.5, 0.04, 'Model quantization strategy', ha='center')

    # Set tick font size
    for label in (axs[1].get_xticklabels() + axs[1].get_yticklabels()):
        label.set_fontsize(14)

    # Show the plot
    plt.show()

    # Print summary table
    if show_table == True:
        table = data.groupby(['quantization strategy'])[['summarization rate (abstracts/min.)', 'model GPU memory footprint (GB)', 'max memory allocated (GB)']].mean().round(2)
        print(table)

    # Return the data
    return data

def batched_summarization_plot(
    data_file: str,
    show_table: bool,
    unique_condition_columns: List[str],
    quantization_method: str,
    oom_columns: List[str], 
    str_columns: List[str], 
    int_columns: List[str], 
    float_columns: List[str],
    oom_replacement_val: Union[str, int, float]) -> pd.DataFrame:

    # Read data
    data = pd.read_csv(data_file)

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

    # Do some unit conversion
    data['summarization rate (abstracts/min.)'] = data['summarization rate (abstracts/sec.)'] * 60
    data['max memory allocated (GB)'] = data['max memory allocated (bytes)'] / 10 ** 9

    # Clean up any leftover NANs
    data.dropna(axis=0, inplace=True)

    # Set general font size
    plt.rcParams['font.size'] = '16'

    # Set figure layout - first column is un-quantized data, second column is quantized data
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.5), tight_layout=True)

    # Split quantized and unquantized data
    unquantized_data = data[data['quantization'] == 'none']
    quantized_data = data[data['quantization'] == quantization_method]

    axs[0].set_title('Summarization rate')
    axs[0].set_xlabel('Batch size')
    axs[0].set_ylabel('abstracts/min.')

    axs[0].errorbar(
        unquantized_data['batch size'].unique(), 
        unquantized_data[['batch size', 'summarization rate (abstracts/min.)']].groupby('batch size').mean()['summarization rate (abstracts/min.)'],
        yerr=unquantized_data[['batch size', 'summarization rate (abstracts/min.)']].groupby('batch size').std()['summarization rate (abstracts/min.)'] * 3,
        capsize=5,
        marker='o', 
        linestyle='none',
        label='unquantized'
    )

    axs[0].errorbar(
        quantized_data['batch size'].unique(), 
        quantized_data[['batch size', 'summarization rate (abstracts/min.)']].groupby('batch size').mean()['summarization rate (abstracts/min.)'],
        yerr=quantized_data[['batch size', 'summarization rate (abstracts/min.)']].groupby('batch size').std()['summarization rate (abstracts/min.)'] * 3,
        capsize=5,
        marker='o', 
        linestyle='none',
        label=quantization_method
    )

    # Set tick font size
    for label in (axs[0].get_xticklabels() + axs[0].get_yticklabels()):
        label.set_fontsize(14)

    # Add legend
    axs[0].legend(loc='upper right', title='Quantization', title_fontsize=14, prop={'size': 12})

    axs[1].set_title('GPU memory use')
    axs[1].set_xlabel('Batch size')
    axs[1].set_ylabel('gigabytes')
    # axs[1].hlines(y=11.4, xmin=-0.95, xmax=8, linewidth=1, color='red')
    # axs[1].hlines(y=3132600320 / 10 ** 9, xmin=-0.95, xmax=8, linewidth=1, color='y')

    axs[1].errorbar(
        unquantized_data['batch size'].unique(), 
        unquantized_data[['batch size', 'max memory allocated (GB)']].groupby('batch size').mean()['max memory allocated (GB)'],
        yerr=unquantized_data[['batch size', 'max memory allocated (GB)']].groupby('batch size').std()['max memory allocated (GB)'] * 3,
        capsize=5,
        marker='o', 
        linestyle='none',
        label='unquantized'
    )

    axs[1].errorbar(
        quantized_data['batch size'].unique(), 
        quantized_data[['batch size', 'max memory allocated (GB)']].groupby('batch size').mean()['max memory allocated (GB)'],
        yerr=quantized_data[['batch size', 'max memory allocated (GB)']].groupby('batch size').std()['max memory allocated (GB)'] * 3,
        capsize=5,
        marker='o', 
        linestyle='none',
        label=quantization_method
    )

    # Set tick font size
    for label in (axs[1].get_xticklabels() + axs[1].get_yticklabels()):
        label.set_fontsize(14)

    # Add legend
    axs[1].legend(loc='lower right', title='Quantization', title_fontsize=14, prop={'size': 12})

    # Draw the plot
    plt.show()

    # Make and print a summary table
    if show_table == True:
        table = data.groupby(['batch size', 'quantization'])[['max memory allocated (GB)', 'summarization rate (abstracts/min.)']].mean().round(1)
        table = table.unstack()
        table = table.astype(str)
        table.replace('<NA>', 'nan', inplace=True)
        print(table)

    return data

def parallel_batched_summarization_plot(
    data_file: str,
    show_table: bool,
    unique_condition_columns: List[str],
    oom_columns: List[str],
    str_columns: List[str],
    int_columns: List[str],
    float_columns: List[str],
    oom_replacement_val: Union[str, int, float]) -> pd.DataFrame:

    data = pd.read_csv(data_file)

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

    data['summarization rate (abstracts/min.)'] = data['summarization rate (abstracts/sec.)'] * 60
    data['max memory allocated (GB)'] = data['max memory allocated (bytes)'] / 1024**3
    data['model memory footprint (GB)'] = data['model memory footprint (bytes)'] / 1024**3

    # Clean up any leftover NANs
    data.dropna(axis=0, inplace=True)

    max_batch_size = max(data['batch size'])
    min_batch_size = min(data['batch size'])
    max_summarization_rate = max(data['summarization rate (abstracts/min.)'])
    min_summarization_rate = min(data['summarization rate (abstracts/min.)'])
    max_memory = max(data['max memory allocated (GB)'])
    min_memory = min(data['max memory allocated (GB)'])
    axis_pad = 0.1

    quantization_types = data['quantization'].unique()

    # Set general font size
    plt.rcParams['font.size'] = '14'

    # Set-up figure and axis array
    fig, axs = plt.subplots(2, 2, figsize=(7, 7), sharex='col', sharey='row', tight_layout=True, gridspec_kw = {'wspace':0, 'hspace':0})

    # Summarization rate plots
    axs_count = 0

    for quantization in quantization_types:
        quantization_type_data = data[data['quantization'] == quantization].copy()
        quantization_type_data.drop('quantization', axis=1, inplace=True)
        worker_nums = quantization_type_data['workers'].unique()

        for workers in worker_nums:

            plot_data = quantization_type_data[quantization_type_data['workers'] == workers]

            mean = plot_data.groupby(['batch size']).mean()
            mean.reset_index(inplace=True)
            
            std = plot_data.groupby(['batch size']).std()
            std.reset_index(inplace=True)

            axs[0, axs_count].set_title(f'Quantization: {quantization}')

            # Only add y axis label on plot in first column
            if axs_count == 0:
                axs[0, axs_count].set_ylabel('Summarization rate\n(abstracts/min.)')

            axs[0, axs_count].errorbar(
                mean['batch size'], 
                mean['summarization rate (abstracts/min.)'], 
                yerr=std['summarization rate (abstracts/min.)'],
                capsize=2.5,
                label=workers,
                linestyle='dotted',
                marker='o'
            )
            
            # Set x axis range
            axs[0, axs_count].set_xlim(min_batch_size - 1, max_batch_size + 1)

            # Set y axis range
            y_range = max_summarization_rate - min_summarization_rate
            axs[0, axs_count].set_ylim(min_summarization_rate - (y_range * 0.1), max_summarization_rate + (y_range * 0.1))

            # Add legend
            axs[0, axs_count].legend(loc='lower right', title='Workers')

            # Set tick font size
            for label in (axs[0, axs_count].get_xticklabels() + axs[0, axs_count].get_yticklabels()):
                label.set_fontsize(14)

        axs_count += 1

    # Memory use plots
    axs_count = 0

    for quantization in quantization_types:
        quantization_type_data = data[data['quantization'] == quantization].copy()
        quantization_type_data.drop('quantization', axis=1, inplace=True)

        for workers in worker_nums:

            plot_data = quantization_type_data[quantization_type_data['workers'] == workers]

            mean = plot_data.groupby(['batch size']).mean()
            mean.reset_index(inplace=True)
            
            std = plot_data.groupby(['batch size']).std()
            std.reset_index(inplace=True)

            axs[1, axs_count].set_xlabel('Batch size')

            # Only add y axis label on plot in first column
            if axs_count == 0:
                axs[1, axs_count].set_ylabel('Max allocated\nmemory (GB)')

            axs[1, axs_count].errorbar(
                mean['batch size'], 
                mean['max memory allocated (GB)'], 
                yerr=std['max memory allocated (GB)'],
                capsize=2.5,
                label=workers,
                linestyle='dotted'
            )

            # axs[1, axs_count].set_xlim([
            #     0,#(min_batch_size - (min_batch_size * axis_pad)), 
            #     (max_batch_size + (max_batch_size * axis_pad))
            # ])
            
            # axs[1, axs_count].set_ylim([
            #     (min_memory - (min_memory * axis_pad)), 
            #     (max_memory + (max_memory * axis_pad))
            # ])

            # Set x axis range
            axs[1, axs_count].set_xlim(min_batch_size - 1, max_batch_size + 1)

            # Set y axis range
            y_range = max_memory - min_memory
            axs[1, axs_count].set_ylim(min_memory - (y_range * 0.1), max_memory + (y_range * 0.1))

            # Add legend
            axs[1, axs_count].legend(loc='lower right', title='Workers')

            # Set tick font size
            for label in (axs[1, axs_count].get_xticklabels() + axs[1, axs_count].get_yticklabels()):
                label.set_fontsize(14)

        axs_count += 1

    # Draw plot
    plt.show()

    # Make and print summary table
    if show_table == True:
        print('Mean max memory allocated (GB)')
        table = data[data['workers'] <= 6].groupby(['workers', 'batch size', 'quantization'])[['max memory allocated (GB)', 'summarization rate (abstracts/min.)']].mean().round(1)
        table = table.unstack()
        table = table.astype(str)
        table.replace('<NA>', 'nan', inplace=True)
        print(table)

    return data

def sql_insert_plot(
    data_file: str,
    show_table: bool) -> pd.DataFrame:

    data = pd.read_csv(data_file)
    data['insert rate (abstracts/millisecond)'] = data['insert rate (abstracts/sec.)'] / 1000
    data['thousand abstracts'] = data['abstracts'] / 1000

    insert_strategies = ['execute_many', 'execute_batch', 'execute_values', 'mogrify', 'stringIO']

    # Set general font size
    plt.rcParams['font.size'] = '16'

    fig, axs = plt.subplots(1, 1, figsize=(5.5, 5.5), tight_layout=True)

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

    # Set tick font size
    for label in (axs.get_xticklabels() + axs.get_yticklabels()):
        label.set_fontsize(14)

    # Add legend
    plt.legend(loc='upper left', prop={'size': 12})

    # Draw plot
    plt.show()

    # Make and print summary table
    if show_table == True:
        print('Mean insert rate (abstracts/millisecond)')
        table = data.groupby(['abstracts', 'insert strategy'])['insert rate (abstracts/millisecond)'].mean().round(2)
        table = table.unstack()
        print(table)

    return data