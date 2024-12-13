import pandas as pd

from pathlib import Path

from data_processing import DataProcessor
from configs import MarketConfig, TradingConfig, DataConfig
from simulator import TradingSimulator
from src.reporting import PerformanceAnalyser, PerformanceVisualiser


def main():
    simulation_timestamp = pd.Timestamp.now()
    # create output directory for results
    results_dir = Path(f'results/sim_{simulation_timestamp}')
    results_dir.mkdir(exist_ok=True)

    # create subdirectory for charts
    plots_dir = results_dir / 'charts'
    plots_dir.mkdir(exist_ok=True)

    # initialise configs
    market_config = MarketConfig()
    trading_config = TradingConfig()
    data_config = DataConfig()

    # load full dataset
    data_processor = DataProcessor('data/data.parquet')
    full_mid_series, full_reference_series, full_spread_series = data_processor.load_and_prepare_data()

    # split data into training and trading portions
    training_size = data_config.TRAINING_POINTS

    training_mid = full_mid_series.iloc[:training_size]
    training_reference = full_reference_series.iloc[:training_size]

    # create and initialise simulation
    simulation = TradingSimulator(
        market_config=market_config,
        trading_config=trading_config,
        data_config=data_config
    )

    # phase 1: Build cycle database
    idx_trading_start = simulation.build_cycle_database(training_mid, training_reference)

    # phase 2: Run trading simulation with online learning
    print("\nStarting trading simulation with online learning...")
    results = simulation.run_sim(full_mid_series, full_reference_series, training_cutoff = idx_trading_start)
    print("\nSimulation Results:")

    # save trades to CSV
    trades_df = pd.DataFrame([{
        'timestamp': t.timestamp,
        'side': t.side.name,
        'size': t.size,
        'price': t.price,
        'fees': t.fees,
        'order_type': t.order_type.name,
        'entry_reason': t.entry_reason.name if t.entry_reason else None,
        'exit_reason': t.exit_reason.name if t.exit_reason else None
    } for t in results['trades']])
    trades_df.to_csv(results_dir / 'simulation_trades.csv', index=False)

    # initialise performance analysers
    perf_analyser = PerformanceAnalyser()
    perf_visualiser = PerformanceVisualiser()

    # calculate P&L and returns
    pnl_series, returns_series = perf_analyser.calculate_pnl(
        results['trades'],
        full_mid_series,
        market_config,
        trading_config
    )
    # generate performance metrics and report
    detailed_report = perf_analyser.generate_report(
        pnl_series,
        returns_series,
        results['trades']
    )
    print(detailed_report)

    # save detailed report
    with open(results_dir / 'detailed_performance_report.txt', 'w') as f:
        f.write(detailed_report)

    # print cycle report
    print(results['performance_report'])

    # create performance visualisations
    perf_visualiser.create_performance_plots(
        pnl=pnl_series,
        trades=results['trades'],
        price_data=pd.DataFrame({'mid': full_mid_series, 'reference': full_reference_series}),
        plot_dir=plots_dir
    )

    print("\nResults have been saved to the 'results' directory:")
    print(f"- Trades data: {results_dir / 'simulation_trades.csv'}")
    print(f"- Detailed performance report: {results_dir / 'detailed_performance_report.txt'}")
    print(f"- Performance plots: {plots_dir / 'performance_plots.png'}")
    print("\nSimulation is complete.")


if __name__ == "__main__":
    main()
