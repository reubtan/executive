import pandas as pd
import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

result_private = pd.read_csv('result_private.csv')

floor_area_ranges = {
    "< 600 sqft": (0, 600),
    "600-850 sqft": (600, 850),
    "850-1100 sqft": (850, 1100),
    "1100-1500 sqft": (1100, 1500),
    "1500 sqft+": (1500, float('inf')),
    "All Resales": (0, float('inf'))
}

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Executive Condominium Resale Gain/Loss Dashboard", style={"textAlign": "center"}),

    # Dropdown for project name
    html.Div([
        html.Label("Select Project Name: (Searchable)"),
        dcc.Dropdown(
            id="project-dropdown",
            options=[{"label": name, "value": name} for name in result_private["project_name"].unique()],
            value=result_private["project_name"].unique()[0],  # Default value
            searchable=True,
            clearable=False
        )
    ], style={"width": "50%", "margin": "0 auto"}),

    # Dropdown for floor area range
    html.Div([
        html.Label("Select Floor Area Range:"),
        dcc.Dropdown(
            id="floor-area-dropdown",
            value="All Resales",  # Default value
            clearable=False
        )
    ], style={"width": "50%", "margin": "0 auto", "marginTop": "10px"}),

    # Display number of entries
    html.Div(id="num-entries", style={"textAlign": "center", "fontSize": "20px", "marginTop": "20px"}),

    # Summary statistics (Largest Gain, Largest Loss, etc.)
    html.Div(id="summary-stats", style={"marginTop": "20px"}),

    # Scatter plot
    dcc.Graph(id="gain-loss-plot")
])

# Callback to update the scatterplot, number of entries, and floor area options
@app.callback(
    [Output("gain-loss-plot", "figure"),
     Output("floor-area-dropdown", "options"),
     Output("floor-area-dropdown", "value"),
     Output("summary-stats", "children")],
    [Input("project-dropdown", "value"),
     Input("floor-area-dropdown", "value")]
)
def update_plot(selected_project, selected_floor_area):
    # Filter data by project name
    filtered_df = result_private[result_private["project_name"] == selected_project]

    # Get unique floor area values for the selected project
    min_area = filtered_df["area_sqft"].min()
    max_area = filtered_df["area_sqft"].max()

    # Filter floor area ranges based on available data
    available_floor_ranges = {key: (min_val, max_val) for key, (min_val, max_val) in floor_area_ranges.items()
                              if min_area <= max_val and max_area >= min_val}

    # Update floor area dropdown options
    floor_area_options = [{"label": key, "value": key} for key in available_floor_ranges.keys()]
    default_floor_area = "All Resales" if "All Resales" in available_floor_ranges else None

    if selected_floor_area is None:
        floor_area_selected = default_floor_area
    else:
        floor_area_selected = selected_floor_area if selected_floor_area in available_floor_ranges else default_floor_area

    # Filter data by floor area range
    floor_min, floor_max = available_floor_ranges.get(floor_area_selected, (0, float('inf')))
    filtered_df = filtered_df[
        (filtered_df["area_sqft"] >= floor_min) & (filtered_df["area_sqft"] <= floor_max)
    ]

    # Calculate the number of entries
    num_entries = len(filtered_df)
    
    filtered_df['sold_at'] = pd.to_datetime(filtered_df['sold_at'])
    filtered_df['held_from'] = pd.to_datetime(filtered_df['held_from'])
    
    # Format the 'sold_at' and 'held_from' columns to only show the date
    filtered_df['Buy Date'] = filtered_df['held_from'].dt.strftime('%Y-%m-%d')
    filtered_df['Sell Date'] = filtered_df['sold_at'].dt.strftime('%Y-%m-%d')
    
    # Extract year sold
    filtered_df['Year Sold'] = filtered_df['sold_at'].dt.year
    filtered_df['Address'] = filtered_df['address']
    filtered_df['Area (sqft)'] = filtered_df['area_sqft']
    
    # Format transaction price and gain/loss with commas
    filtered_df['Transacted Price (SGD)'] = filtered_df['transaction_price_dollars'].apply(lambda x: f"{x:,.0f}")
    filtered_df['Area (sqft)'] = filtered_df['area_sqft'].apply(lambda x: f"{x:,.0f}")
    filtered_df['Gain/Loss (SGD)'] = filtered_df['Gain/Loss'].apply(lambda x: f"{x:,.0f}")
    
    filtered_df['sold_at'] = pd.to_datetime(filtered_df['sold_at'])
    filtered_df['held_from'] = pd.to_datetime(filtered_df['held_from'])
    filtered_df['Year-Month Sold'] = filtered_df['sold_at'].dt.to_period('M').astype(str)
    filtered_df['Gain/Loss Category'] = filtered_df['Gain/Loss'].apply(
        lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Zero')
    )

    # Calculate holding period statistics
    if not filtered_df.empty:
        max_months = filtered_df["months_held"].max()
        median_months = filtered_df["months_held"].median()
        min_months = filtered_df["months_held"].min()
    else:
        max_months = median_months = min_months = 0

    # Convert months to "years and months" format
    def format_years_months(total_months):
        years = total_months // 12
        months = total_months % 12
        return f"{years} years {months} months" if years > 0 else f"{months} months"

    max_period = format_years_months(max_months)
    median_period = format_years_months(int(median_months))  # Ensure median is an integer
    min_period = format_years_months(min_months)

    num_gains = (filtered_df['Gain/Loss'] > 0).sum()
    num_losses = (filtered_df['Gain/Loss'] < 0).sum()
    num_zero = (filtered_df['Gain/Loss'] == 0).sum()    
    # Calculate Gain/Loss statistics
    largest_gain = filtered_df[filtered_df['Gain/Loss'] > 0]['Gain/Loss'].max() if not filtered_df[filtered_df['Gain/Loss'] > 0].empty else 0
    largest_loss = filtered_df[filtered_df['Gain/Loss'] < 0]['Gain/Loss'].min() if not filtered_df[filtered_df['Gain/Loss'] < 0].empty else 0
    median_gain = filtered_df[filtered_df['Gain/Loss'] > 0]['Gain/Loss'].median() if not filtered_df[filtered_df['Gain/Loss'] > 0].empty else 0
    median_loss = filtered_df[filtered_df['Gain/Loss'] < 0]['Gain/Loss'].median() if not filtered_df[filtered_df['Gain/Loss'] < 0].empty else 0
    smallest_gain = filtered_df[filtered_df['Gain/Loss'] > 0]['Gain/Loss'].min() if not filtered_df[filtered_df['Gain/Loss'] > 0].empty else 0
    smallest_loss = filtered_df[filtered_df['Gain/Loss'] < 0]['Gain/Loss'].max() if not filtered_df[filtered_df['Gain/Loss'] < 0].empty else 0

    total_gains = (filtered_df['Gain/Loss'] > 0).sum()
    total_losses = (filtered_df['Gain/Loss'] < 0).sum()
    total_resales = len(filtered_df)
    percentage_gains = total_gains / total_resales * 100
    percentage_loss = total_losses / total_resales * 100
    percentage_zero = 100 - (percentage_gains + percentage_loss)
    
    # Summary statistics
    summary_stats = html.Div([
        html.Hr(),
        html.Div(f"Number of Resales: {num_entries}", style={"marginBottom": "5px"}),
        html.Div(f"Gains: {num_gains} ({percentage_gains:.2f}%)", style={"marginBottom": "5px"}),
        html.Div(f"Losses: {num_losses} ({percentage_loss:.2f}%)", style={"marginBottom": "5px"}),
        html.Div(f"Sold at same price: {num_zero} ({percentage_zero:.2f}%)", style={"marginBottom": "5px"}),
        html.Hr(),
        html.Div(f"Max Holding Period: {max_period}", style={"marginBottom": "5px"}),
        html.Div(f"Median Holding Period: {median_period}", style={"marginBottom": "5px"}),
        html.Div(f"Min Holding Period: {min_period}", style={"marginBottom": "5px"}),
        html.Hr(),

        # Grouping Largest Gain/Loss, Median Gain/Loss, Smallest Gain/Loss
        html.Div(f"Median Gain/Loss: {filtered_df['Gain/Loss'].median():,.0f} SGD", style={"marginBottom": "5px"}),
        html.Div([
            html.Div(f"Largest Gain: {largest_gain:,.0f} SGD", style={"width": "45%", "display": "inline-block"}),
            html.Div(f"Largest Loss: {largest_loss:,.0f} SGD", style={"width": "45%", "display": "inline-block"})
        ], style={"marginBottom": "5px"}),

        html.Div([
            html.Div(f"Median Gain: {median_gain:,.0f} SGD", style={"width": "45%", "display": "inline-block"}),
            html.Div(f"Median Loss: {median_loss:,.0f} SGD", style={"width": "45%", "display": "inline-block"})
        ], style={"marginBottom": "5px"}),

        html.Div([
            html.Div(f"Smallest Gain: {smallest_gain:,.0f} SGD", style={"width": "45%", "display": "inline-block"}),
            html.Div(f"Smallest Loss: {smallest_loss:,.0f} SGD", style={"width": "45%", "display": "inline-block"})
        ], style={"marginBottom": "5px"}),

        html.Hr()  # Add a line after Smallest Gain/Smallest Loss
    ], style={"textAlign": "center", "fontSize": "18px"})



    # Define the color mapping for categories
    color_map = {
        'Positive': 'green',
        'Negative': 'red',
        'Zero': 'yellow'
    }

    # Create scatter plot
    fig = px.scatter(
        filtered_df,
        x="Year-Month Sold",
        y="Gain/Loss",
        hover_data={
            "Gain/Loss (SGD)": True,
            "Buy Date": True,
            "Sell Date": True,
            "Area (sqft)": True,
            "Address": True,
            "Transacted Price (SGD)": True,
            "Gain/Loss": False,
            "Gain/Loss Category": False
        },
        color="Gain/Loss Category",
        color_discrete_map=color_map
    )

    fig.update_layout(
        transition_duration=500,
        height=800,
        yaxis=dict(
            tickformat=',.0f'
        )
    )

    return fig, floor_area_options, floor_area_selected, summary_stats


if __name__ == "__main__":
    app.run_server(port=8052, debug=True)
