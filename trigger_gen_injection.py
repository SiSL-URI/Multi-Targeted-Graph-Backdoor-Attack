import torch
import numpy as np
import random
import networkx as nx


def trigger_gen_injection(train_graphs, test_graphs0, test_graphs1, test_graphs2, frac, num_backdoor_nodes, seed, 
     target_label, prob, num_trigger, node_feat_size, num_classes):

    # Generating the trigger
    print(train_graphs[0])

    G_gens = []
    G_gen_feats = []
    attacking_node_index = []
    G_edge_attrs = []

    for target_idx in range(len(target_label)):
        ## erdos_renyi
        G_gen = []
        G_gen_feat = []  # (num_backdoor_nodes, node_feat_size) for each trigger
        G_edge_attr = []
        
        
        for i in range(num_trigger):
            G = nx.erdos_renyi_graph(num_backdoor_nodes, prob, seed = target_idx*1000+i*100+seed)
            G_gen.append(G)
            
            rng = np.random.RandomState(target_idx*1000 + i*100 + seed)
            x_part  = rng.uniform(low=0.0, high=1.0, size=(num_backdoor_nodes, 3))
            pos_part = rng.uniform(low=0.0, high=1.0, size=(num_backdoor_nodes, 2))
            trig_feat_np = np.hstack([x_part, pos_part])  
            trig_feat = torch.from_numpy(trig_feat_np).float()            # torch.FloatTensor
            G_gen_feat.append(trig_feat)

            torch.manual_seed(target_idx*1000 + i*100 + seed)  
            temp = torch.FloatTensor(len(G.edges)+1).uniform_(0.0, 1.0)
            temp_doubled = temp.repeat_interleave(2)
            G_edge_attr.append(temp_doubled)
    
        G_gens.append(G_gen)
        G_gen_feats.append(G_gen_feat)
        G_edge_attrs.append(G_edge_attr)

        
        #selecting the attacking node
        np.random.seed(target_idx*100 + seed) 
        attacking_nodes = np.random.choice(num_backdoor_nodes, num_trigger)
        attacking_node_index.append(attacking_nodes)

    

    
    # Selecting the graphs which will be backdoored

    label_indices = [[] for _ in range(num_classes)]
    num_labelwise_train_graphs = []

    for idx in range(len(train_graphs)):
        label_indices[train_graphs[idx].y].append(idx)

    for i in range (num_classes):
        num_labelwise_train_graphs.append(len(label_indices[i]))


    used_index = []
    rand_backdoor_graph_idx = []

    for target_idx in range(len(target_label)):
        train_backdoor_graphs_indexes = []
        for i in range (num_classes):
            if i == target_label[target_idx]:
                continue
            else:
                random.seed(i*100+seed)
                num_backdoor_train_graphs = int(num_labelwise_train_graphs[i]*frac)
                available_indices = [idx for idx in label_indices[i] if idx not in used_index]
                print(f'available in {target_idx}th target for class {i}: {len(available_indices)}')
                labelwise_idx_temp = random.sample(available_indices,
                                                k=num_backdoor_train_graphs)  # without replacement
        
                train_backdoor_graphs_indexes.append(labelwise_idx_temp)
        
        flattened = [item for sublist in train_backdoor_graphs_indexes for item in sublist]
        rand_backdoor_graph_idx.append(flattened)
        used_index.extend(flattened)

    for i in range(len(rand_backdoor_graph_idx)):
         print(f'number of backdoored graphs in train_graphs for target label {target_label[i]} = {len(rand_backdoor_graph_idx[i])}')
        



    data_test = train_graphs[0]
    print("Check")
    print(data_test)
    #Trigger injection to the Training Dataset

    data_test1 = train_graphs[0]
    print("Check graph inside func")
    print(data_test1)
    #Trigger injection to the Training Dataset

    for target_idx in range(len(target_label)):
            
        # for loop for selecting the node which will be targeted in a graph 
        for i in range(num_trigger):
            for idx in rand_backdoor_graph_idx[target_idx]:
                num_nodes = len(train_graphs[idx].x)
                np.random.seed(i*100+seed)
                rand_select_nodes = np.random.choice(num_nodes, 1)
        
                edges = train_graphs[idx].edge_index.transpose(1, 0).numpy().tolist()
        
                for e in G_gens[target_idx][i].edges:
                    edges.append([int(num_nodes) + e[0], int(num_nodes)+e[1]])
                    edges.append([int(num_nodes) + e[1], int(num_nodes)+e[0]])
        
                edges.append([rand_select_nodes[0], int(num_nodes)+attacking_node_index[target_idx][i]])
                edges.append([int(num_nodes)+attacking_node_index[target_idx][i], rand_select_nodes[0]])
                

                data = train_graphs[idx]
                if not isinstance(data.x, torch.Tensor):
                    data.x = torch.tensor(data.x, dtype=torch.float)
                
                # Use torch.cat instead of np.concatenate
                data.x = torch.cat([data.x, G_gen_feats[target_idx][i]], dim=0)
                data.edge_index =  torch.LongTensor(np.asarray(edges).transpose())
                data.edge_attr = torch.cat([data.edge_attr, G_edge_attrs[target_idx][i]], dim=0)
                data.y = torch.tensor([target_label[target_idx]], dtype=torch.long)
                train_graphs._data_list[idx] = data


    # Test graph with 1st target
    for i in range(num_trigger):
        test_graphs_targetlabel_indexes = []
        test_backdoor_graphs_indexes = []
        for graph_idx in range(len(test_graphs0)):
            if test_graphs0[graph_idx].y != target_label[0]:
                test_backdoor_graphs_indexes.append(graph_idx)
            else:
                test_graphs_targetlabel_indexes.append(graph_idx)
        print('#test target label:', len(test_graphs_targetlabel_indexes), '#test backdoor labels:',
              len(test_backdoor_graphs_indexes))
        
    
        for idx in test_backdoor_graphs_indexes:
            num_nodes = len(test_graphs0[idx].x)
            rand_select_nodes = np.random.choice(num_nodes, 1)
    
            edges = test_graphs0[idx].edge_index.transpose(1, 0).numpy().tolist()
      
            for e in G_gens[0][i].edges:
                # print([rand_select_nodes[e[0]], rand_select_nodes[e[1]]])
                edges.append([int(num_nodes) + e[0], int(num_nodes)+e[1]])
                edges.append([int(num_nodes) + e[1], int(num_nodes)+e[0]])
    
            edges.append([rand_select_nodes[0], int(num_nodes)+attacking_node_index[0][i]])
            edges.append([int(num_nodes)+attacking_node_index[0][i], rand_select_nodes[0]])


            data = test_graphs0[idx]
            if not isinstance(data.x, torch.Tensor):
                data.x = torch.tensor(data.x, dtype=torch.float)
            
            data.x = torch.cat([data.x, G_gen_feats[0][i]], dim=0)
            data.edge_index =  torch.LongTensor(np.asarray(edges).transpose())
            data.edge_attr = torch.cat([data.edge_attr, G_edge_attrs[0][i]], dim=0)
            test_graphs0._data_list[idx] = data
            
    test_backdoor_graphs0 = [graph for graph in test_graphs0 if graph.y != target_label[0]]



    # Test graph with 2rd target
    #Change the test_graph name
    #Change test_backdoor_graphs0 to test_backdoor_graphs1
    tagert_label_idx = 1
    
    for i in range(num_trigger):
        test_graphs_targetlabel_indexes = []
        test_backdoor_graphs_indexes = []
        for graph_idx in range(len(test_graphs1)):
            if test_graphs1[graph_idx].y != target_label[tagert_label_idx]:
                test_backdoor_graphs_indexes.append(graph_idx)
            else:
                test_graphs_targetlabel_indexes.append(graph_idx)
        print('#test target label:', len(test_graphs_targetlabel_indexes), '#test backdoor labels:',
              len(test_backdoor_graphs_indexes))
        
    
        for idx in test_backdoor_graphs_indexes:
            num_nodes = len(test_graphs1[idx].x)
            rand_select_nodes = np.random.choice(num_nodes, 1)
    
            edges = test_graphs1[idx].edge_index.transpose(1, 0).numpy().tolist()
      
            for e in G_gens[tagert_label_idx][i].edges:
                # print([rand_select_nodes[e[0]], rand_select_nodes[e[1]]])
                edges.append([int(num_nodes) + e[0], int(num_nodes)+e[1]])
                edges.append([int(num_nodes) + e[1], int(num_nodes)+e[0]])
    
            edges.append([rand_select_nodes[0], int(num_nodes)+attacking_node_index[tagert_label_idx][i]])
            edges.append([int(num_nodes)+attacking_node_index[tagert_label_idx][i], rand_select_nodes[0]])


            data = test_graphs1[idx]
            if not isinstance(data.x, torch.Tensor):
                data.x = torch.tensor(data.x, dtype=torch.float)
                
            data.x = torch.cat([data.x, G_gen_feats[tagert_label_idx][i]], dim=0)
            data.edge_index =  torch.LongTensor(np.asarray(edges).transpose())
            data.edge_attr = torch.cat([data.edge_attr, G_edge_attrs[tagert_label_idx][i]], dim=0)
            test_graphs1._data_list[idx] = data
            
    test_backdoor_graphs1 = [graph for graph in test_graphs1 if graph.y != target_label[tagert_label_idx]]


    # Test graph with 3rd target
    #Change the test_graph name
    #Change test_backdoor_graphs1 to test_backdoor_graphs2
    tagert_label_idx = 2
    
    for i in range(num_trigger):
        test_graphs_targetlabel_indexes = []
        test_backdoor_graphs_indexes = []
        for graph_idx in range(len(test_graphs2)):
            if test_graphs2[graph_idx].y != target_label[tagert_label_idx]:
                test_backdoor_graphs_indexes.append(graph_idx)
            else:
                test_graphs_targetlabel_indexes.append(graph_idx)
        print('#test target label:', len(test_graphs_targetlabel_indexes), '#test backdoor labels:',
              len(test_backdoor_graphs_indexes))
        
    
        for idx in test_backdoor_graphs_indexes:
            num_nodes = len(test_graphs2[idx].x)
            rand_select_nodes = np.random.choice(num_nodes, 1)
    
            edges = test_graphs2[idx].edge_index.transpose(1, 0).numpy().tolist()
      
            for e in G_gens[tagert_label_idx][i].edges:
                # print([rand_select_nodes[e[0]], rand_select_nodes[e[1]]])
                edges.append([int(num_nodes) + e[0], int(num_nodes)+e[1]])
                edges.append([int(num_nodes) + e[1], int(num_nodes)+e[0]])
    
            edges.append([rand_select_nodes[0], int(num_nodes)+attacking_node_index[tagert_label_idx][i]])
            edges.append([int(num_nodes)+attacking_node_index[tagert_label_idx][i], rand_select_nodes[0]])


            data = test_graphs2[idx]
            if not isinstance(data.x, torch.Tensor):
                data.x = torch.tensor(data.x, dtype=torch.float)
                
            data.x = torch.cat([data.x, G_gen_feats[tagert_label_idx][i]], dim=0)
            data.edge_index =  torch.LongTensor(np.asarray(edges).transpose())
            data.edge_attr = torch.cat([data.edge_attr, G_edge_attrs[tagert_label_idx][i]], dim=0)
            test_graphs2._data_list[idx] = data
            
    test_backdoor_graphs2 = [graph for graph in test_graphs2 if graph.y != target_label[tagert_label_idx]]



    
    return train_graphs, test_backdoor_graphs0, test_backdoor_graphs1, test_backdoor_graphs2