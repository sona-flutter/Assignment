import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
warnings.filterwarnings('ignore')

class SalesAnalysis:
    def __init__(self):
        """Initialize paths and create necessary directories"""
        # Get the current directory where the script is located
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define paths relative to the base directory
        self.data_path = os.path.join(self.base_dir, 'data', 'AusApparalSales4thQrt2020.csv')
        self.figures_path = os.path.join(self.base_dir, 'reports', 'figures')
        self.report_path = os.path.join(self.base_dir, 'reports')
        
        self.df = None
        self.state_wise = None
        self.group_wise = None
        self.time_wise = None
        
        # Create directories if they don't exist
        os.makedirs(os.path.join(self.base_dir, 'data'), exist_ok=True)
        os.makedirs(self.figures_path, exist_ok=True)
        os.makedirs(self.report_path, exist_ok=True)
        
        print("Initialized SalesAnalysis with directories:")
        print(f"Data path: {self.data_path}")
        print(f"Figures path: {self.figures_path}")
        print(f"Report path: {self.report_path}")
        
    def load_data(self):
        """Load and clean the CSV data"""
        try:
            if not os.path.exists(self.data_path):
                print(f"Error: CSV file not found at {self.data_path}")
                return False
                
            self.df = pd.read_csv(self.data_path)
            print("\nData loaded successfully!")
            print("\nDataset Info:")
            print(f"Rows: {self.df.shape[0]}")
            print(f"Columns: {self.df.shape[1]}")
            print("\nColumns in dataset:")
            print(self.df.columns.tolist())
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def check_missing_values(self):
        """Check for missing values in the dataset"""
        print("\nChecking for missing values:")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Calculate percentage of missing values
        missing_percentage = (missing_values / len(self.df)) * 100
        print("\nPercentage of missing values:")
        print(missing_percentage[missing_percentage > 0])
        
        return missing_values > 0
        
    def clean_data(self):
        """Clean the data by handling missing values and normalization"""
        print("\nStarting data cleaning process...")
        
        # Print available columns
        print("\nAvailable columns in dataset:")
        print(self.df.columns.tolist())
        
        # Identify numeric columns
        numeric_columns = ['Unit', 'Sales']
        print("\nNumeric columns to be normalized:")
        print(numeric_columns)
        
        # Handle missing values
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if col in numeric_columns:
                    # Fill numeric columns with mean
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                else:
                    # Fill categorical columns with mode
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        # Normalize numerical columns
        if numeric_columns:
            scaler = MinMaxScaler()
            self.df[numeric_columns] = scaler.fit_transform(self.df[numeric_columns])
        
        print("Data cleaning completed!")
        print("\nVerifying no missing values remain:")
        print(self.df.isnull().sum())

    def perform_analysis(self):
        """Perform basic statistical analysis"""
        print("\nPerforming data analysis...")
        
        # Group data by State
        self.state_wise = self.df.groupby('State')['Sales'].sum().reset_index()
        print("\nTop 3 Performing States:")
        print(self.state_wise.nlargest(3, 'Sales'))
        print("\nBottom 3 Performing States:")
        print(self.state_wise.nsmallest(3, 'Sales'))
        
        # Group analysis
        self.group_wise = self.df.groupby('Group')['Sales'].sum().reset_index()
        print("\nGroup Performance:")
        print(self.group_wise.sort_values('Sales', ascending=False))
        
        # Time analysis
        if 'Time' in self.df.columns:
            self.time_wise = self.df.groupby('Time')['Sales'].mean().reset_index()
            print("\nPeak Sales Times:")
            print(self.time_wise.nlargest(3, 'Sales'))
        
        # Calculate descriptive statistics
        stats = self.df[['Unit', 'Sales']].describe()
        print("\nDescriptive Statistics:")
        print(stats)
        
        return stats

    def create_visualizations(self):
        """Create various visualizations for the analysis"""
        print("\nGenerating visualizations...")
        
        # 1. State-wise Sales Bar Plot
        plt.figure(figsize=(12, 6))
        if self.state_wise is not None:
            state_data = self.state_wise.sort_values('Sales', ascending=False)
            plt.bar(state_data['State'], state_data['Sales'])
            plt.title('State-wise Sales Analysis')
            plt.xlabel('State')
            plt.ylabel('Sales')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_path, 'state_wise_sales.png'))
            plt.close()
        
        # 2. Group Analysis Pie Chart
        plt.figure(figsize=(10, 10))
        group_data = self.df.groupby('Group')['Sales'].sum()
        plt.pie(group_data, labels=group_data.index, autopct='%1.1f%%')
        plt.title('Sales Distribution by Group')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, 'group_sales.png'))
        plt.close()
        
        # 3. Sales Distribution Box Plot
        plt.figure(figsize=(12, 6))
        plt.boxplot([self.df[self.df['State'] == state]['Sales'] 
                    for state in self.df['State'].unique()],
                   labels=self.df['State'].unique())
        plt.title('Sales Distribution by State')
        plt.xlabel('State')
        plt.ylabel('Sales')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, 'sales_distribution.png'))
        plt.close()
        
        # 4. Time Series Analysis
        if 'Time' in self.df.columns:
            plt.figure(figsize=(12, 6))
            time_sales = self.df.groupby('Time')['Sales'].mean()
            plt.plot(time_sales.index, time_sales.values)
            plt.title('Average Sales Throughout the Day')
            plt.xlabel('Time')
            plt.ylabel('Average Sales')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_path, 'time_of_day_sales.png'))
            plt.close()
        
        # 5. Unit vs Sales Scatter Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['Unit'], self.df['Sales'], alpha=0.5)
        plt.title('Unit vs Sales Relationship')
        plt.xlabel('Units')
        plt.ylabel('Sales')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, 'unit_sales_relationship.png'))
        plt.close()
        
        print("All visualizations created and saved in:", self.figures_path)

    def generate_report(self):
   
        print("\nGenerating analysis report...")
        
        report = ["# AAL Sales Analysis Report\n"]
        report.append("## Executive Summary")
        report.append("This report analyzes the sales data for AAL's fourth quarter, providing insights into state-wise and group-wise performance.\n")
        
        report.append("## Data Overview")
        report.append(f"- Total Records: {len(self.df)}")
        report.append(f"- Time Period: Fourth Quarter")
        report.append(f"- States Analyzed: {len(self.df['State'].unique())}")
        report.append(f"- Customer Groups: {len(self.df['Group'].unique())}\n")
        
        report.append("## Key Findings")
        
        report.append("### State Performance")
        report.append("#### Top Performing States:")
        top_states = self.state_wise.nlargest(3, 'Sales')
        report.append("| State | Sales |")
        report.append("|-------|-------|")
        for _, row in top_states.iterrows():
            report.append(f"| {row['State']} | {row['Sales']:.2f} |")
        
        report.append("\n#### Lowest Performing States:")
        bottom_states = self.state_wise.nsmallest(3, 'Sales')
        report.append("| State | Sales |")
        report.append("|-------|-------|")
        for _, row in bottom_states.iterrows():
            report.append(f"| {row['State']} | {row['Sales']:.2f} |")
        
        report.append("\n### Customer Group Analysis")
        report.append("Group-wise sales distribution:")
        report.append("| Group | Sales |")
        report.append("|-------|-------|")
        group_sales = self.group_wise.sort_values('Sales', ascending=False)
        for _, row in group_sales.iterrows():
            report.append(f"| {row['Group']} | {row['Sales']:.2f} |")
        
        report.append("\n## Recommendations")
        report.append("1. Focus Areas:")
        report.append("   - Implement targeted marketing in low-performing states")
        report.append("   - Develop special promotions for underperforming customer groups")
        report.append("   - Optimize inventory based on state-wise demand\n")
        
        report.append("2. Growth Strategies:")
        report.append("   - Analyze and replicate successful practices from top-performing states")
        report.append("   - Develop customer retention programs")
        report.append("   - Enhance online sales channels\n")
        
        report.append("## Statistical Analysis")
        report.append("### Sales Statistics")
        stats = self.df['Sales'].describe()
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        for metric, value in stats.items():
            report.append(f"| {metric} | {value:.2f} |")
        
        report.append("\n## Visualizations")
        report.append("The following visualizations have been generated and saved in the 'figures' folder:")
        report.append("1. State-wise Sales Analysis")
        report.append("2. Sales Distribution by Group")
        report.append("3. Sales Distribution by State")
        report.append("4. Time of Day Analysis")
        report.append("5. Unit vs Sales Relationship")
        
        report_content = "\n".join(report)
        report_file = os.path.join(self.report_path, 'sales_report.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print("Report generated and saved as:", report_file)

def main():
    # Create instance of SalesAnalysis
    analysis = SalesAnalysis()
    
    # Execute all analysis steps
    if analysis.load_data():
        analysis.check_missing_values()
        analysis.clean_data()
        analysis.perform_analysis()
        analysis.create_visualizations()
        analysis.generate_report()
        print("\nAnalysis completed successfully!")
        print("Please check the reports folder for the detailed report and visualizations.")
    else:
        print("Analysis stopped due to data loading error.")
        print("Please ensure the CSV file is in the correct location and try again.")

if __name__ == "__main__":
    main()