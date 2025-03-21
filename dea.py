import os
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url
import math
import time
import plotly.graph_objects as go
# Import funkcie 'dea' zo správneho modulu
from dealib.dea.core._dea import dea
import numpy as np

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.QUARTZ, dbc.icons.FONT_AWESOME, dbc_css], suppress_callback_exceptions=True)


# First, update the load_data function to accept a wave parameter
def load_data(wave=None):
    try:
        if wave:
            # Extract wave number (e.g., "1 vlna" -> "1")
            wave_num = wave.split()[0]
            file_path = f'data/merged_data_{wave_num}.xlsx'
            print(f"Loading wave data from: {file_path}")
            df = pd.read_excel(file_path)
        else:
            # Default behavior (load the original file)
            df = pd.read_csv('data/nmerged_data.csv')
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        # Try to find any CSV in the data folder as fallback
        data_dir = 'data'
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(data_dir, file))
                    break
        else:
            # Return empty DataFrame if no data found
            return pd.DataFrame({'No data found': ['Create data folder with CSV files']})
    
    # Add Projekt column if the DataFrame is not empty
    if not df.empty and 'Projekt' not in df.columns:
        # Create project names: project_1, project_2, etc.
        df.insert(0, 'Projekt', [f"project_{i+1}" for i in range(len(df))])
    
    return df

# Load data once at startup
df = load_data()



# Define app layout
app.layout = html.Div([
    dcc.Store(id="selected-rows-store", data=[]),
    dcc.Store(id="filtered-data-store", data=df.to_dict('records')),
    dcc.Store(id="current-sort", data={"column": None, "direction": None}),
    dcc.Store(id="select-all-action", data={"action": None, "timestamp": 0}),  # Add this line here
    dcc.Store(id="dea-results-store", data=None),  # Add this to your app.layout after the other Store components


    dbc.Container([
        # Add this at the beginning of your app layout, right after the app title
        dbc.Row([
            dbc.Col([
                html.H1("Data Explorer App", className="mt-4 mb-4"),
            ], width=9),
            dbc.Col([
                dbc.Label("Select Data Wave:"),
                dbc.Select(
                    id="wave-selector",
                    options=[
                        {"label": "1 vlna", "value": "1 vlna"},
                        {"label": "2 vlna", "value": "2 vlna"},
                        {"label": "3 vlna", "value": "3 vlna"},
                    ],
                    value=None,
                    className="mb-3"
                ),
            ], width=3, className="d-flex flex-column justify-content-end"),
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H3("Source Data"),
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Reload Data", id="reload-button", color="primary", className="me-2"),
                        dbc.Button("Unselect All", id="unselect-button", color="danger", className="me-2"),
                        dbc.Button("Random Select", id="random-select-button", color="info", className="me-2"),  # Add this line
                        dbc.Input(
                            id='global-search',
                            placeholder='Search across all columns...',
                            type='text',
                            className="",
                            style={'width': '300px'}
                        ),
                    ], width=12, className="d-flex align-items-center mb-3")
                ]),
                
                # Source data pagination controls
                dbc.Row([
                    dbc.Col([
                        dbc.InputGroup([
                            dbc.InputGroupText("Page Size:"),
                            dbc.Select(
                                id="page-size-dropdown",
                                options=[
                                    {"label": "10 rows", "value": "10"},
                                    {"label": "25 rows", "value": "25"},
                                    {"label": "50 rows", "value": "50"},
                                    {"label": "100 rows", "value": "100"},
                                ],
                                value="10",
                                style={"width": "150px"}
                            ),
                            dbc.InputGroupText("Page:"),
                            dbc.Select(
                                id="page-number-dropdown",
                                options=[{"label": str(i+1), "value": str(i)} for i in range(math.ceil(len(df)/10))],
                                value="0",
                                style={"width": "100px"}
                            )
                        ], className="mb-3"),
                    ], width=6),
                    dbc.Col([
                        html.Div(id="page-info", className="text-end pt-2")
                    ], width=6),
                ], className="mb-2"),
                
                html.Div(id="source-table-container", 
                    style={
                        "max-height": "500px",
                        "overflow-y": "auto",
                        "position": "relative"  # For fixed header positioning
                    },
                    children=[
                        dbc.Table(
                            # Header
                            [html.Thead(html.Tr(
                                [html.Th(html.Div([
                                    dbc.Checkbox(id="select-all-checkbox"),
                                ]), style={"width": "50px"})] + 
                                [html.Th(col, className="sortable", id={"type": "sort-column", "column": col}) for col in df.columns]
                            ), style={"position": "sticky", "top": 0, "z-index": 1, "background-color": "#fff"})]+
                            # Body - will be filled by callback
                            [html.Tbody(id="source-table-body")],
                            bordered=True,
                            hover=True,
                            responsive=True,
                            striped=True,
                            id="source-table",
                            className="mb-4"
                        ),
                    ]
                ),
            ], width=12),
        ], className="mb-2"),
        
        dbc.Row([
            dbc.Col([
                html.H3("Selected Data"),
                # Selected Data Table
                html.Div(id="selected-table-container",
                    style={
                        "max-height": "500px",
                        "overflow-y": "auto",
                        "position": "relative"  # For fixed header positioning
                    }, 
                    children=[
                        dbc.Table(
                            # Header
                            [html.Thead(html.Tr([
                                html.Th(col) for col in df.columns
                            ]))]+
                            # Body - will be filled by callback
                            [html.Tbody(id="selected-table-body")],
                            bordered=True,
                            hover=True,
                            responsive=True,
                            id="selected-table",
                            className="mb-4"
                        ),
                ]),
            ], width=12),
        ], className="mb-2"),
        
        # Store components to track state
        # dcc.Store(id="selected-rows-store", data=[]),
        # dcc.Store(id="filtered-data-store", data=df.to_dict('records')),
        # dcc.Store(id="current-sort", data={"column": None, "direction": None}),
    ]),

    # Add this after your existing layout sections, before the closing brackets
    dbc.Row([
        dbc.Col([
            html.H3("Create DEA Frontier Graph", className="mt-4 container"),
            dbc.Card([
                dbc.CardBody([
                    html.H5("Output Selection", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Select Output Columns:"),
                            dbc.Checklist(
                                id="output-checklist",
                                options=[
                                    {"label": "O1", "value": "O1"},
                                    {"label": "O2", "value": "O2"},
                                    {"label": "O3", "value": "O3"},
                                ],
                                value=["O1"],  # Default selection
                                inline=True,
                                className="mb-3"
                            ),
                        ], width=8),
                        dbc.Col([
                            dbc.Button("Create Graph", id="create-graph-button", color="success", className="mt-4"),
                        ], width=4, className="d-flex justify-content-end"),
                    ]),
                    dcc.Graph(id="dea-graph", className="mt-3", style={"height": "600px"}),
                ])
            ]),
        ], width=12),
    ]),

    # Add this right after the existing DEA graph section
    dbc.Row([
        dbc.Col([
            html.H3("Create 3D DEA Visualization", className="mt-4 container"),
            dbc.Card([
                dbc.CardBody([
                    html.H5("Output Combination Selection", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Select Output Axes:"),
                            dbc.RadioItems(
                                id="output-3d-choice",
                                options=[
                                    {"label": "O1 × O2", "value": "O1_O2"},
                                    {"label": "O1 × O3", "value": "O1_O3"},
                                    {"label": "O2 × O3", "value": "O2_O3"},
                                ],
                                value="O1_O2",  # Default selection
                                inline=True,
                                className="mb-3"
                            ),
                        ], width=8),
                        dbc.Col([
                            dbc.Button("Create 3D Graph", id="create-3d-button", color="success", className="mt-4"),
                        ], width=4, className="d-flex justify-content-end"),
                    ]),
                    dcc.Graph(id="dea-3d-graph", className="mt-3", style={"height": "600px"}),
                ])
            ]),
        ], width=12),
    ]),

])

# Keep only this callback which includes selected_rows
@app.callback(
    [Output("source-table-body", "children"),
     Output("page-info", "children"),
     Output("page-number-dropdown", "options")],
    [Input("filtered-data-store", "data"),
     Input("page-size-dropdown", "value"),
     Input("page-number-dropdown", "value"),
     Input("current-sort", "data"),
     Input("selected-rows-store", "data")]  # Added selected rows as input
)
def update_source_table(data, page_size, page_number, sort_data, selected_rows):
    # Rest of the function remains the same
    if not data:
        return [], "No data available", []
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Apply sorting if specified
    if sort_data and sort_data["column"]:
        ascending = sort_data["direction"] == "asc"
        df = df.sort_values(by=sort_data["column"], ascending=ascending)
    
    # Calculate pagination
    page_size = int(page_size)
    page_number = int(page_number)
    total_pages = math.ceil(len(df) / page_size)
    
    # Update page dropdown options
    page_options = [{"label": str(i+1), "value": str(i)} for i in range(total_pages)]
    
    # Slice the data for the current page
    start_idx = page_number * page_size
    end_idx = min(start_idx + page_size, len(df))
    page_data = df.iloc[start_idx:end_idx]
    
    # Create table rows with checkboxes
    rows = []
    for i, (_, row) in enumerate(page_data.iterrows()):
        row_idx = start_idx + i
        is_selected = row_idx in selected_rows if selected_rows else False
        
        checkbox = html.Td(dbc.Checkbox(
            id={"type": "row-checkbox", "index": row_idx},
            className="row-checkbox",
            value=is_selected  # Set checkbox state based on selection
        ))
        
        cells = [checkbox] + [html.Td(row[col]) for col in df.columns]
        rows.append(html.Tr(
            cells, 
            id={"type": "table-row", "index": row_idx},
            className="selected-row" if is_selected else ""  # Optional: add class for styling
        ))
    
    # Page info text
    page_info = f"Showing {start_idx + 1} to {end_idx} of {len(df)} entries"
    
    return rows, page_info, page_options
#
# 1. Modify this callback to break the dependency on selected-rows-store
@app.callback(
    Output("select-all-checkbox", "value"),
    [Input("page-size-dropdown", "value"),
     Input("page-number-dropdown", "value")],
    [State("filtered-data-store", "data"),
     State("selected-rows-store", "data")]
)
def update_select_all_state(page_size, page_number, filtered_data, selected_rows):
    if not filtered_data:
        return False
    
    # Check if all items on current page are selected
    page_size = int(page_size)
    page_number = int(page_number)
    start_idx = page_number * page_size
    end_idx = min(start_idx + page_size, len(filtered_data))
    
    # Get indices of current page
    current_page_indices = list(range(start_idx, end_idx))
    
    # If there are no selected rows, checkbox should be unchecked
    if not selected_rows:
        return False
    
    # Check if all rows on current page are selected
    all_selected = all(idx in selected_rows for idx in current_page_indices)
    
    return all_selected

# Then update the callback to include wave-selector as an input
# Fix the filtered data callback
@app.callback(
    [Output("filtered-data-store", "data"),
     Output("selected-rows-store", "data", allow_duplicate=True),
     Output("page-number-dropdown", "value")],
    [Input("wave-selector", "value"),
     Input("reload-button", "n_clicks"),
     Input("global-search", "value")],
    prevent_initial_call=True  # Changed from False to True
)
def update_filtered_data(selected_wave, n_clicks, search_term):
    ctx = callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Load data based on the selected wave
    if trigger == "wave-selector" or trigger == "reload-button" or trigger is None:
        df = load_data(selected_wave)
    else:
        # For search, use current data
        df = load_data(selected_wave)
    
    # Apply search filter if provided
    if search_term:
        # Create a filter mask that checks all columns
        filter_mask = pd.Series(False, index=df.index)
        for col in df.columns:
            # Convert column to string and check if it contains the search term (case insensitive)
            filter_mask = filter_mask | df[col].astype(str).str.contains(search_term, case=False, na=False)
        
        # Apply the filter
        filtered_df = df[filter_mask]
        return filtered_df.to_dict('records'), [], "0"  # Reset selected rows and page
    
    return df.to_dict('records'), [], "0"  # Reset selected rows and page
# Add this to your dcc.Store components
# Store components to track state

# Modified row selection callback with random selection by region
@app.callback(
    Output("selected-rows-store", "data", allow_duplicate=True),
    [Input({"type": "row-checkbox", "index": dash.dependencies.ALL}, "value"),
     Input("unselect-button", "n_clicks"),
     Input("random-select-button", "n_clicks"),
     Input("select-all-action", "data")],
    [State({"type": "row-checkbox", "index": dash.dependencies.ALL}, "id"),
     State("filtered-data-store", "data"),
     State("page-size-dropdown", "value"),
     State("page-number-dropdown", "value"),
     State("selected-rows-store", "data")],
    prevent_initial_call=True  # Changed from False to True
)
def update_selected_rows(checked_values, unselect_clicks, random_select_clicks, select_all_action, 
                         checkbox_ids, filtered_data, page_size, page_number, current_selected):
    ctx = callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle unselect all button
    if trigger == "unselect-button":
        return []
    
    # Handle random select by region button (NEW FUNCTIONALITY)
    if trigger == "random-select-button":
        try:
            # Convert to DataFrame
            df = pd.DataFrame(filtered_data)
            
            # Check if 'Kraj' column exists
            if 'Kraj' not in df.columns:
                print("Error: 'Kraj' column not found in data")
                return current_selected  # Keep current selection if no Kraj column
            
            # Get unique values in the 'Kraj' column
            unique_regions = df['Kraj'].unique()
            print(f"Found {len(unique_regions)} unique regions: {unique_regions}")
            
            # Initialize list to store selected indices
            selected_indices = []
            
            # For each unique region, randomly select 3 rows
            for region in unique_regions:
                # Get indices of rows for this region
                region_rows = df[df['Kraj'] == region]
                
                if len(region_rows) > 0:
                    # Determine how many samples to take (min of 3 or available rows)
                    n_samples = min(3, len(region_rows))
                    
                    # Select random rows from this region
                    selected_rows = region_rows.sample(n=n_samples)
                    
                    # Find these rows in the original filtered_data list
                    for idx in selected_rows.index:
                        # Map DataFrame index to position in filtered_data
                        position = list(df.index).index(idx)
                        selected_indices.append(position)
            
            print(f"Selected {len(selected_indices)} rows across {len(unique_regions)} regions")
            return selected_indices
        
        except Exception as e:
            print(f"Error in random selection by region: {str(e)}")
            import traceback
            traceback.print_exc()
            return current_selected  # Keep current selection if error occurs
    
    # Handle select all action
    if trigger == "select-all-action":
        select_all = select_all_action.get("action")
        if select_all:
            # Calculate current page indices
            page_size = int(page_size)
            page_number = int(page_number)
            start_idx = page_number * page_size
            end_idx = min(start_idx + page_size, len(filtered_data))
            
            # Add all indices from current page
            return current_selected + list(range(start_idx, end_idx))
        elif select_all is False:  # Explicitly False, not None
            # Remove all indices from current page
            page_size = int(page_size)
            page_number = int(page_number)
            start_idx = page_number * page_size
            end_idx = min(start_idx + page_size, len(filtered_data))
            
            return [idx for idx in current_selected if idx < start_idx or idx >= end_idx]
    
    # Handle individual checkboxes
    if "row-checkbox" in trigger:
        selected_indices = []
        
        for i, checked in enumerate(checked_values):
            if checked:
                row_idx = checkbox_ids[i]["index"]
                selected_indices.append(row_idx)
        
        return selected_indices
    
    return current_selected
# 2. Add this callback for the select-all checkbox to directly update selections
@app.callback(
    Output("select-all-action", "data"),
    [Input("select-all-checkbox", "value")],
    [State("select-all-action", "data"),
     State("page-size-dropdown", "value"),
     State("page-number-dropdown", "value"),
     State("filtered-data-store", "data")]
)
def update_select_all_action(select_all, current, page_size, page_number, filtered_data):
    # This callback is only triggered by direct user interaction with the checkbox
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
        
    if current is None:
        current = {"action": None, "timestamp": 0}
    
    # Only update if the value actually changed
    if select_all != current.get("action"):
        # Add page information to the action so we know which rows to select/deselect
        return {
            "action": select_all, 
            "timestamp": time.time(),
            "page_size": int(page_size),
            "page_number": int(page_number),
            "page_count": len(filtered_data) if filtered_data else 0
        }
    
    return dash.no_update
# Callback to update the selected table
# Update the selected table callback to show DEA results if available
@app.callback(
    Output("selected-table-body", "children"),
    [Input("selected-rows-store", "data"),
     Input("filtered-data-store", "data"),
     Input("dea-results-store", "data")]  # Add this input
)
def update_selected_table(selected_indices, filtered_data, dea_results):
    print(len(selected_indices), len(filtered_data), len(dea_results) if dea_results else 0)
    print("tototoototoot")
    if not selected_indices or not filtered_data:
        return []
    
    # If DEA results are available, use those instead of filtered data
    if dea_results is not None:
        # Use the DEA processed data
        rows = []
        df = pd.DataFrame(dea_results)
        for _, row in df.iterrows():
            cells = [html.Td(row[col]) for col in df.columns]
            rows.append(html.Tr(cells))
        
        return rows
    
    # Otherwise use the original data
    df = pd.DataFrame(filtered_data)
    selected_data = [filtered_data[i] for i in selected_indices if i < len(filtered_data)]
    selected_df = pd.DataFrame(selected_data)
    
    if selected_df.empty:
        return []
    
    # Create table rows
    rows = []
    for _, row in selected_df.iterrows():
        cells = [html.Td(row[col]) for col in df.columns]
        rows.append(html.Tr(cells))
    
    return rows

# Also update the header of the selected table to include the new columns
@app.callback(
    Output("selected-table-container", "children"),
    [Input("dea-results-store", "data"),
     Input("filtered-data-store", "data")]
)
def update_selected_table_header(dea_results, filtered_data):
    if dea_results is not None:
        # Get columns from DEA results
        columns = pd.DataFrame(dea_results).columns
    else:
        # Get columns from filtered data
        columns = pd.DataFrame(filtered_data).columns
    
    return html.Div([
        dbc.Table(
            [html.Thead(html.Tr([
                html.Th(col) for col in columns
            ]))]+
            [html.Tbody(id="selected-table-body")],
            bordered=True,
            hover=True,
            responsive=True,
            id="selected-table",
            className="mb-4"
        ),
    ])

# Callback to handle column sorting
@app.callback(
    Output("current-sort", "data"),
    [Input({"type": "sort-column", "column": dash.dependencies.ALL}, "n_clicks")],
    [State("current-sort", "data"),
     State({"type": "sort-column", "column": dash.dependencies.ALL}, "id")]
)
def update_sort(n_clicks, current_sort, column_ids):
    if not n_clicks or not any(n_clicks):
        return current_sort
    
    ctx = callback_context
    if not ctx.triggered:
        return current_sort
    
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    clicked_col = eval(trigger)["column"]
    
    # Toggle sort direction or set new column
    if current_sort["column"] == clicked_col:
        new_direction = "desc" if current_sort["direction"] == "asc" else "asc"
        return {"column": clicked_col, "direction": new_direction}
    else:
        return {"column": clicked_col, "direction": "asc"}


def calc_dea(mel_df):
    # Definujeme vstupy – stĺpec 'I1'
    inputs = mel_df[['I1']].values

    # Definujeme výstupy – stĺpce 'O1', 'O2', 'O3'
    outputs = mel_df[['O1', 'O2', 'O3']].values

    # Vyriešime DEA problém s outputovou orientáciou a RTS nastaveným na "vrs" (pre BCC model)
    eff = dea(x=inputs, y=outputs, rts="vrs", orientation="output")

    # Priradíme vypočítané efektívnostné skóre do pôvodného DataFrame
    mel_df['efficiency'] = eff.eff

    # Pridáme nový stĺpec 'is_efficient' – DMU je efektívna, ak je efficiency <= 1, inak 0
    mel_df['is_efficient'] = mel_df['efficiency'].apply(lambda x: 1 if x <= 1 else 0)

    # Vypíšeme aktualizovaný DataFrame s novými stĺpcami
    print("DataFrame s priradenými hodnotami efektívnosti a is_efficient:")
    print(mel_df[['I1', 'O1', 'O2', 'O3', 'efficiency', 'is_efficient']])
    return mel_df

# Update the create_dea_graph callback to also update the dea-results-store
@app.callback(
    [Output("dea-graph", "figure"),
     Output("dea-results-store", "data")],  # Add this output
    [Input("create-graph-button", "n_clicks")],
    [State("selected-rows-store", "data"),
     State("filtered-data-store", "data"),
     State("output-checklist", "value")],
    prevent_initial_call=True
)
def create_dea_graph(n_clicks, selected_indices, filtered_data, selected_outputs):
    empty_fig = go.Figure().update_layout(title="No data selected")
    
    if not n_clicks or not selected_indices or not filtered_data or not selected_outputs:
        # Return empty figure if no data or no outputs selected
        return empty_fig, None
    
    # Create DataFrame from selected rows
    selected_data = [filtered_data[i] for i in selected_indices if i < len(filtered_data)]
    if not selected_data:
        return empty_fig, None
    
    df = pd.DataFrame(selected_data)

    # Rename columns
    df = df.rename(columns={
        'final_51510': 'O1',
        'final_52100': 'O2',
        'final_54000': 'O3',
        'Rozpočet': 'I1',
    })

    # Apply DEA calculations
    df = calc_dea(df)


    # Make sure the required columns exist
    required_cols = ['I1'] + selected_outputs
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        return go.Figure().update_layout(title=f"Missing columns: {', '.join(missing)}"), None
    
    # Determine output option based on checkbox selection
    output_option = "single" if len(selected_outputs) == 1 else "sum"
    output_col = selected_outputs[0] if len(selected_outputs) == 1 else None
    
    # Return both the figure and the processed dataframe
    return (dea_frontier_plotly(
        df, 
        output_col=output_col, 
        output_option=output_option, 
        agg_outputs=selected_outputs,
        title=f"DEA Frontier: I1 vs {' + '.join(selected_outputs)}"
    ), df.to_dict('records'))
# Update the dea_frontier_plotly function to include hovering project names
def dea_frontier_plotly(df, output_col=None, output_option="single", agg_outputs=None, title=None):
    """Plotly version of the DEA frontier visualization"""
    if agg_outputs is None:
        agg_outputs = ["O1", "O2", "O3"]
    
    # Decide on the output column based on output_option
    if output_option == "mean":
        df = df.copy()
        df["O_mean"] = df[agg_outputs].mean(axis=1)
        used_output = "O_mean"
    elif output_option == "sum":
        df = df.copy()
        df["O_sum"] = df[agg_outputs].sum(axis=1)
        used_output = "O_sum"
    elif output_option == "single":
        if output_col is None:
            raise ValueError("When using output_option 'single', you must specify an output_col.")
        used_output = output_col
    else:
        raise ValueError("output_option must be one of: 'single', 'mean', or 'sum'.")
    
    # Compute summary statistics
    minx = df['I1'].min()
    maxx = df['I1'].max()
    miny = df[used_output].min()
    maxy = df[used_output].max()
    
    # Separate efficient and non-efficient DMU
    df_eff = df[df['is_efficient'] == 1].copy()
    df_ineff = df[df['is_efficient'] == 0].copy()
    
    # Construct CRS Frontier (output-oriented)
    best_ratio = 0
    best_x, best_y = 0, 0
    for _, row in df.iterrows():
        xi = row['I1']
        yi = row[used_output]
        if xi > 0:
            ratio = yi / xi
            if ratio > best_ratio:
                best_ratio = ratio
                best_x = xi
                best_y = yi
    crs_x = [0, best_x]
    crs_y = [0, best_y]
    
    # Construct VRS Frontier
    df_eff_sorted = df_eff.sort_values(by='I1')
    vrs_x = list(df_eff_sorted['I1'])
    vrs_y = list(df_eff_sorted[used_output])
    
    # Create a polygon for filling
    poly_x = [minx] + vrs_x + [maxx, minx]
    poly_y = [miny] + vrs_y + [miny, miny]
    
    # Create the Plotly figure
    fig = go.Figure()
    
    # Add efficient area fill
    fig.add_trace(go.Scatter(
        x=poly_x, y=poly_y,
        fill='toself',
        fillcolor='rgba(255, 165, 0, 0.15)',
        line=dict(color='rgba(255, 165, 0, 0)'),
        name='Efficient area',
        showlegend=True,
        hoverinfo='skip'  # No hover for the area fill
    ))
    
    # Add VRS frontier line
    fig.add_trace(go.Scatter(
        x=vrs_x, y=vrs_y,
        mode='lines+markers',
        line=dict(color='black', dash='dash'),
        name='VRS frontier',
        hoverinfo='skip'  # No hover for the frontier line
    ))
    
    # Add CRS frontier line
    fig.add_trace(go.Scatter(
        x=crs_x, y=crs_y,
        mode='lines',
        line=dict(color='orange'),
        name='CRS frontier',
        hoverinfo='skip'  # No hover for the frontier line
    ))
    
    # Check for Projekt column
    has_projekt = 'Projekt' in df.columns
    
    # Add efficient DMU points with hover text
    if not df_eff.empty:
        hover_text = df_eff['Projekt'] if has_projekt else None
        
        fig.add_trace(go.Scatter(
            x=df_eff['I1'], 
            y=df_eff[used_output],
            mode='markers',
            marker=dict(color='blue', size=12),
            name='Efficient DMU',
            text=hover_text,  # Add project names as hover text
            hovertemplate='<b>%{text}</b><br>' +
                          'Input (I1): %{x}<br>' +
                          f'Output ({used_output}): %{{y}}<br>' +
                          'Efficiency: 1.0<extra></extra>' if has_projekt else None
        ))
    
    # Add non-efficient DMU points with hover text
    if not df_ineff.empty:
        hover_text = df_ineff['Projekt'] if has_projekt else None
        efficiency_values = df_ineff['efficiency'].round(3)
        
        fig.add_trace(go.Scatter(
            x=df_ineff['I1'], 
            y=df_ineff[used_output],
            mode='markers',
            marker=dict(color='gray', size=8),
            name='Not-efficient DMU',
            text=hover_text,  # Add project names as hover text
            customdata=efficiency_values,  # Add efficiency scores for hover
            hovertemplate='<b>%{text}</b><br>' +
                          'Input (I1): %{x}<br>' +
                          f'Output ({used_output}): %{{y}}<br>' +
                          'Efficiency: %{customdata}<extra></extra>' if has_projekt else None
        ))
    
    # Update layout
    fig.update_layout(
        title=title or f"DEA Output-Oriented Frontier: I1 vs {used_output}",
        xaxis_title="Input (I1)",
        yaxis_title=f"Output ({used_output})",
        xaxis=dict(range=[0, maxx * 1.1]),
        yaxis=dict(range=[0, maxy * 1.1]),
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    return fig

# Add this after your existing callbacks
@app.callback(
    Output("dea-3d-graph", "figure"),
    [Input("create-3d-button", "n_clicks")],
    [State("dea-results-store", "data"),
     State("output-3d-choice", "value")],
    prevent_initial_call=True
)
def create_dea_3d_graph(n_clicks, dea_results, output_choice):
    # Default empty figure
    empty_fig = go.Figure().update_layout(
        title="No DEA results available",
        annotations=[{
            'text': "Run DEA analysis first by selecting data and clicking 'Create Graph'",
            'showarrow': False,
            'font': {'size': 16}
        }]
    )
    
    if not n_clicks or not dea_results:
        return empty_fig
    
    try:
        # Convert results to DataFrame
        df = pd.DataFrame(dea_results)
        
        # Verify the necessary columns exist
        required_cols = ['I1', 'O1', 'O2', 'O3', 'is_efficient']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            return go.Figure().update_layout(
                title=f"Missing required columns: {', '.join(missing)}",
                annotations=[{
                    'text': "The DEA results must include Input, Outputs and efficiency data",
                    'showarrow': False,
                    'font': {'size': 14}
                }]
            )
        
        # Determine output columns based on selection
        if output_choice == "O1_O2":
            output_col1, output_col2 = "O1", "O2"
            title = "DEA 3D Visualization: I1 vs O1 and O2"
        elif output_choice == "O1_O3":
            output_col1, output_col2 = "O1", "O3"
            title = "DEA 3D Visualization: I1 vs O1 and O3"
        elif output_choice == "O2_O3":
            output_col1, output_col2 = "O2", "O3"
            title = "DEA 3D Visualization: I1 vs O2 and O3"
        else:
            output_col1, output_col2 = "O1", "O2"  # Default
            title = "DEA 3D Visualization"
        
        # Separate efficient and non-efficient DMU
        df_eff = df[df['is_efficient'] == 1].copy()
        df_ineff = df[df['is_efficient'] == 0].copy()
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add project names if available
        project_col = 'Projekt' if 'Projekt' in df.columns else None
        
        # Efficient DMU: blue markers with project labels
        fig.add_trace(
            go.Scatter3d(
                x=df_eff['I1'],
                y=df_eff[output_col1],
                z=df_eff[output_col2],
                mode='markers+text' if project_col else 'markers',
                marker=dict(size=8, color='blue'),
                text=df_eff[project_col] if project_col else None,
                textposition="top center",
                name='Efficient DMU'
            )
        )
        
        # Not-efficient DMU: gray markers with project labels
        if not df_ineff.empty:
            fig.add_trace(
                go.Scatter3d(
                    x=df_ineff['I1'],
                    y=df_ineff[output_col1],
                    z=df_ineff[output_col2],
                    mode='markers+text' if project_col else 'markers',
                    marker=dict(size=6, color='gray'),
                    text=df_ineff[project_col] if project_col else None,
                    textposition="top center",
                    name='Not-efficient DMU'
                )
            )
        
        # Configure layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='I1 (Input)',
                yaxis_title=f'{output_col1} (Output)',
                zaxis_title=f'{output_col2} (Output)'
            ),
            legend=dict(x=0, y=1.0),
            template="plotly_white"
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in 3D graph generation: {str(e)}")
        return go.Figure().update_layout(
            title="Error generating 3D graph",
            annotations=[{
                'text': str(e),
                'showarrow': False,
                'font': {'size': 14, 'color': 'red'}
            }]
        )
  

# Remove or comment out your existing update_filtered_data callback as it will conflict
     
if __name__ == '__main__':
    app.run(debug=True, port=8050)