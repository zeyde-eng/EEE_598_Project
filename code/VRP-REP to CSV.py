import xml.etree.ElementTree as ET
import pandas as pd

def convert_vrp_xml_to_csv(xml_file_path, output_csv_path=None):
    """
    Convert VRP-REP XML instance to CSV format.
    
    Parameters:
    xml_file_path (str): Path to the XML file
    output_csv_path (str): Path for output CSV file (optional)
    
    Returns:
    tuple: (nodes_df, fleet_df, requests_df) - DataFrames containing the parsed data
    """
    
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Extract instance info
    info = root.find('info')
    instance_name = info.find('name').text if info.find('name') is not None else 'Unknown'
    dataset = info.find('dataset').text if info.find('dataset') is not None else 'Unknown'
    
    print(f"Processing instance: {instance_name} from dataset: {dataset}")
    
    # Extract nodes data
    nodes_data = []
    network = root.find('network')
    nodes = network.find('nodes')
    
    for node in nodes.findall('node'):
        node_id = int(node.get('id'))
        node_type = int(node.get('type'))  # 0 = depot, 1 = customer
        cx = float(node.find('cx').text)
        cy = float(node.find('cy').text)
        
        nodes_data.append({
            'node_id': node_id,
            'type': 'depot' if node_type == 0 else 'customer',
            'x_coord': cx,
            'y_coord': cy
        })
    
    nodes_df = pd.DataFrame(nodes_data)
    
    # Extract fleet data
    fleet_data = []
    fleet = root.find('fleet')
    if fleet is not None:
        for vehicle_profile in fleet.findall('vehicle_profile'):
            profile_type = vehicle_profile.get('type')
            departure_node = int(vehicle_profile.find('departure_node').text)
            arrival_node = int(vehicle_profile.find('arrival_node').text)
            # Handle capacity as float first, then convert to int
            capacity = int(float(vehicle_profile.find('capacity').text))
            
            fleet_data.append({
                'profile_type': profile_type,
                'departure_node': departure_node,
                'arrival_node': arrival_node,
                'capacity': capacity
            })
        
        # Count number of vehicles if specified
        vehicle_count = len(fleet.findall('vehicle'))
        if vehicle_count == 0:
            # If no explicit vehicles, extract from instance name (e.g., A-n32-k5 means 5 vehicles)
            if '-k' in instance_name:
                vehicle_count = int(instance_name.split('-k')[1])
        
        if fleet_data:
            fleet_data[0]['vehicle_count'] = vehicle_count
    
    fleet_df = pd.DataFrame(fleet_data) if fleet_data else pd.DataFrame()
    
    # Extract requests/demands data
    requests_data = []
    requests = root.find('requests')
    if requests is not None:
        for request in requests.findall('request'):
            request_id = int(request.get('id'))
            node_id = int(request.get('node'))
            
            # Check for deterministic quantity
            quantity_elem = request.find('quantity')
            if quantity_elem is not None:
                demand = float(quantity_elem.text)
                demand_type = 'deterministic'
                distribution = None
                parameters = None
            else:
                # Check for uncertain/stochastic quantity
                uncertain_qty = request.find('uncertain_quantity')
                if uncertain_qty is not None:
                    random_var = uncertain_qty.find('random_variable')
                    distribution = random_var.get('distribution')
                    demand_type = 'stochastic'
                    
                    # Extract parameters
                    params = {}
                    for param in random_var.findall('parameter'):
                        param_name = param.get('name')
                        param_value = float(param.text)
                        params[param_name] = param_value
                    
                    parameters = str(params)
                    # Use appropriate parameter as expected demand
                    demand = params.get('lambda', params.get('mean', params.get('mu', 0)))
                else:
                    demand = 0
                    demand_type = 'none'
                    distribution = None
                    parameters = None
            
            requests_data.append({
                'request_id': request_id,
                'node_id': node_id,
                'demand': demand,
                'demand_type': demand_type,
                'distribution': distribution,
                'parameters': parameters
            })
    
    requests_df = pd.DataFrame(requests_data) if requests_data else pd.DataFrame()
    
    # Create combined CSV output
    if output_csv_path:
        combined_data = []
        
        for _, node in nodes_df.iterrows():
            node_id = node['node_id']
            
            # Find corresponding request/demand
            demand_info = requests_df[requests_df['node_id'] == node_id]
            if len(demand_info) > 0:
                demand_row = demand_info.iloc[0]
                demand = demand_row['demand']
                demand_type = demand_row['demand_type']
                distribution = demand_row['distribution']
                parameters = demand_row['parameters']
            else:
                demand = 0 if node['type'] == 'customer' else 0
                demand_type = 'none'
                distribution = None
                parameters = None
            
            combined_data.append({
                'instance_name': instance_name,
                'dataset': dataset,
                'node_id': node_id,
                'node_type': node['type'],
                'x_coord': node['x_coord'],
                'y_coord': node['y_coord'],
                'demand': demand,
                'demand_type': demand_type,
                'distribution': distribution,
                'parameters': parameters
            })
        
        combined_df = pd.DataFrame(combined_data)
        combined_df.to_csv(output_csv_path, index=False)
        print(f"Combined data saved to: {output_csv_path}")
        
        # Save fleet information separately
        if not fleet_df.empty:
            fleet_csv_path = output_csv_path.replace('.csv', '_fleet.csv')
            fleet_df['instance_name'] = instance_name
            fleet_df.to_csv(fleet_csv_path, index=False)
            print(f"Fleet data saved to: {fleet_csv_path}")
    
    return nodes_df, fleet_df, requests_df

# Usage example
if __name__ == "__main__":
    # Convert the XML file to CSV
    nodes_df, fleet_df, requests_df = convert_vrp_xml_to_csv('..\\data\\christiansen-and-lysgaard-2007\\E-n51-k5.xml', '..\\data\\new_dataset\\E-n51-k5.csv')
    
    print("\nConversion Summary:")
    print(f"Nodes: {len(nodes_df)} (including {len(nodes_df[nodes_df['type'] == 'depot'])} depot(s))")
    print(f"Customer nodes: {len(nodes_df[nodes_df['type'] == 'customer'])}")
    print(f"Requests: {len(requests_df)}")
    print(f"Fleet profiles: {len(fleet_df)}")
    
    # Display sample data
    print("\nSample nodes:")
    print(nodes_df.head())
    
    if not fleet_df.empty:
        print("\nFleet information:")
        print(fleet_df)
    
    if not requests_df.empty:
        print("\nSample requests/demands:")
        print(requests_df.head())
