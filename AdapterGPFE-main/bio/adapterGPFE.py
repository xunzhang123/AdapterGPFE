import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from loader import BioDataset
from dataloader import DataLoaderFinetune
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): Dimensionality of embeddings for nodes and edges.
        input_layer (bool): Whether the GIN conv is applied to the input layer or not.
                             (Input node labels are uniform...)

    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add", input_layer=False):
        super(GINConv, self).__init__(aggr)

        self.edge_encoder = torch.nn.Linear(9, emb_dim)

        # Multi-layer perceptron (MLP) for node aggregation
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_dim, 2 * emb_dim),
            torch.nn.BatchNorm1d(2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim)
        )

        # Uniform input layer (optional)
        self.input_layer = input_layer
        if self.input_layer:
            self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
            torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

        # Initialize placeholders for embeddings
        self.embeded_x = None
        self.aggr_x = None

    def forward(self, x, edge_index, edge_attr, prompt=None):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(1))

        # Add features for self-loop edges
        self_loop_attr = torch.zeros(x.size(0), 9, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 7] = 1
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # Compute edge embeddings
        edge_embeddings = self.edge_encoder(edge_attr)

        # Apply input layer transformation if enabled
        if self.input_layer:
            x = self.input_node_embeddings(x.to(torch.int64).view(-1, ))
            if prompt is not None:
                x = prompt.add(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings), x, self.aggr_x

    def message(self, x_j, edge_attr):
        return torch.cat([x_j, edge_attr], dim=1)

    def update(self, aggr_out):
        self.aggr_x = aggr_out
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr="add", input_layer=False):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)

        ### Mapping 0/1 edge features to embedding
        self.edge_encoder = torch.nn.Linear(9, emb_dim)

        ### Mapping uniform input features to embedding.
        self.input_layer = input_layer
        if self.input_layer:
            self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
            torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 9)
        self_loop_attr[:, 7] = 1  # attribute for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_encoder(edge_attr)

        if self.input_layer:
            x = self.input_node_embeddings(x.to(torch.int64).view(-1, ))

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add", input_layer=False):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        ### Mapping 0/1 edge features to embedding
        self.edge_encoder = torch.nn.Linear(9, heads * emb_dim)

        ### Mapping uniform input features to embedding.
        self.input_layer = input_layer
        if self.input_layer:
            self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
            torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 9)
        self_loop_attr[:, 7] = 1  # attribute for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_encoder(edge_attr)

        if self.input_layer:
            x = self.input_node_embeddings(x.to(torch.int64).view(-1, ))

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean", input_layer=False):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)

        ### Mapping 0/1 edge features to embedding
        self.edge_encoder = torch.nn.Linear(9, emb_dim)

        ### Mapping uniform input features to embedding.
        self.input_layer = input_layer
        if self.input_layer:
            self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
            torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 9)
        self_loop_attr[:, 7] = 1  # attribute for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_encoder(edge_attr)

        if self.input_layer:
            x = self.input_node_embeddings(x.to(torch.int64).view(-1, ))

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(torch.nn.Module):

    def __init__(self, args, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin", max_bottleneck_dim=15, min_bottleneck_dim=1):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        # 保存传入的参数
        self.max_bottleneck_dim = max_bottleneck_dim
        self.min_bottleneck_dim = min_bottleneck_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of message-passing GNN convs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if layer == 0:
                input_layer = True
            else:
                input_layer = False

            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add", input_layer=input_layer))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim, input_layer=input_layer))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim, input_layer=input_layer))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim, input_layer=input_layer))

        prompt_num = 2

        gating = 0.01
        self.gating_parameter = torch.nn.Parameter(torch.zeros(prompt_num, num_layer, 1))
        self.gating_parameter.data += gating
        self.register_parameter('gating_parameter', self.gating_parameter)
        self.gating = self.gating_parameter

        self.prompts = torch.nn.ModuleList()
        for i in range(prompt_num):
            self.prompts.append(torch.nn.ModuleList())

            bottleneck_dim = self.compute_bottleneck_dim(layer, num_layer)

        for layer in range(num_layer):
            for i in range(prompt_num):
                if bottleneck_dim > 0:
                    self.prompts[i].append(torch.nn.Sequential(
                        torch.nn.Linear(2 * emb_dim if i > 0 else emb_dim, bottleneck_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(bottleneck_dim, emb_dim),
                        torch.nn.BatchNorm1d(emb_dim)
                    ))
                    torch.nn.init.zeros_(self.prompts[i][-1][2].weight.data)
                    torch.nn.init.zeros_(self.prompts[i][-1][2].bias.data)
                else:
                    self.prompts[i].append(torch.nn.BatchNorm1d(emb_dim))


    def compute_bottleneck_dim(self, layer, total_layers):
        """
        Compute bottleneck dimension for a given layer based on its position.
        Layers closer to the output have larger bottleneck_dim.
        """

        return int(self.min_bottleneck_dim + (self.max_bottleneck_dim - self.min_bottleneck_dim) * (
                    layer / (total_layers - 1)))

    def forward(self, x, edge_index, edge_attr, prompt):
        h_list = [x]
        for layer in range(self.num_layer):
            h = h_list[layer]

            h, x_embeded, x_aggr = self.gnns[layer](h, edge_index, edge_attr, prompt)

            delta = self.prompts[0][layer](x_embeded)
            h = h + delta * self.gating[0][layer]
            delta = self.prompts[1][layer](x_aggr)
            h = h + delta * self.gating[1][layer]

            if layer < self.num_layer - 1:
                h = F.relu(h)
            h = F.dropout(h, self.drop_ratio, training=self.training)

            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list[1:], dim=0), dim=0)[0]

        return node_representation



class AdapterGPFE_graphpred(torch.nn.Module):


    def __init__(self, args, num_layer, emb_dim, num_tasks, JK="last", drop_ratio=0, graph_pooling="mean",
                 gnn_type="gin", max_bottleneck_dim=15, min_bottleneck_dim=1):
        super(AdapterGPFE_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(args, num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type, max_bottleneck_dim=max_bottleneck_dim, min_bottleneck_dim=min_bottleneck_dim)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)
    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage), strict=False)

    def forward(self, data, prompt):
        x, edge_index, edge_attr, batch, prompt = data.x, data.edge_index, data.edge_attr, data.batch, prompt
        node_representation = self.gnn(x, edge_index, edge_attr, prompt)

        pooled = self.pool(node_representation, batch)
        center_node_rep = node_representation[data.center_node_idx]

        graph_rep = torch.cat([pooled, dim=1)

        return self.graph_pred_linear(graph_rep)


if __name__ == "__main__":
    pass