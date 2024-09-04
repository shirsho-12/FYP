import dgl
import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.utils import convert
from typing import Optional, Tuple, Union
from transformers import BertModel
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    QuestionAnsweringModelOutput,
)
from project.consts.constants import dicto


class GraphQA(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        # Hyper Parameters
        self.num_labels = config.num_labels
        self.node_types = ["token", "leaf", "constituent"]
        self.metadata = (
            # Node
            ["token", "leaf", "constituent"],
            # Edge
            [
                ("token", "connect", "token"),
                ("constituent", "connect", "constituent"),
                ("constituent", "rev_connect", "constituent"),
                ("constituent", "connect", "token"),
                ("token", "rev_connect", "constituent"),
            ],
        )
        self.graph_hidden_channels = 768
        self.number_of_constituents = 82
        self.input_shape = 768
        self.graph_layer = 2
        self.graph_head = 2

        # BERT Backbone
        # TODO: Swap backbones
        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Heterogenous Graph
        self.lin_dict = nn.ModuleDict()

        self.lin_dict["token"] = nn.Linear(
            config.hidden_size, self.graph_hidden_channels
        )
        self.lin_dict["constituent"] = nn.Linear(
            self.number_of_constituents, self.graph_hidden_channels
        )

        self.convs = nn.ModuleList()
        for _ in range(self.graph_layer):
            conv = HGTConv(
                self.input_shape,
                self.graph_hidden_channels,
                self.metadata,
                self.graph_head,
                group="sum",
            )
            self.convs.append(conv)

        self.graph_qa_outputs = nn.Linear(self.graph_hidden_channels, 1)
        self.graph2_qa_outputs = nn.Linear(self.graph_hidden_channels, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        sep_index: Optional[torch.Tensor] = None,
        graph_data: Optional[dict] = None,
        return_dict: Optional[bool] = None,
        test: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """

        input_ids.to("cuda:0")
        attention_mask.to("cuda:0")
        token_type_ids.to("cuda:0")

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        data = HeteroData()
        data["constituent"].node_id = graph_data[0]["graph"].to_dict()["constituent"][
            "node_id"
        ]
        data["constituent"].x = graph_data[0]["graph"].to_dict()["constituent"]["x"]
        data["constituent"].y = graph_data[0]["graph"].to_dict()["constituent"]["y"]
        data["token"].node_id = torch.arange(outputs[0].size(1))
        data["token"].x = outputs[0][0]
        transform = T.Compose([T.AddSelfLoops(), T.ToUndirected()])
        transform2 = T.AddSelfLoops()
        transform1 = T.RemoveIsolatedNodes()

        data["constituent", "connect", "constituent"].edge_index = (
            graph_data[0]["graph"]
            .to_dict()["_global_store"]["('constituent', 'connect', 'constituent')"]
            .t()
        )
        data["constituent", "connect", "token"].edge_index = (
            graph_data[0]["graph"]
            .to_dict()["_global_store"]["('constituent', 'connect', 'token')"]
            .t()
        )
        data.to("cpu")

        datai = transform1(data)
        g = convert.to_dgl(datai)

        datai = datai.to("cpu")
        data2 = datai.to_homogeneous()
        c = convert.to_dgl(data2)

        y = graph_data[0]["graph"].to_dict()["constituent"]["y"].float()

        kop = y
        lp_type = None
        if torch.argmax(y).item() != 0 and end_positions.item() != 0:
            mop = []

            lp = torch.argmax(data["constituent"].x[torch.argmax(y).item()])
            for key, value in dicto.items():
                if lp.item() == value:
                    lp_type = key
            for ter in data["constituent"].x:
                if torch.equal(ter, data["constituent"].x[torch.argmax(y).item()]):
                    mop.append(1)
                else:
                    mop.append(0)
            kop = torch.tensor(mop)

            result = torch.cat(
                dgl.traversal.bfs_nodes_generator(c, torch.argmax(y).item())
            )
            msk = result.ge(g.num_nodes("constituent"))
            tok = torch.masked_select(result, msk)
            lm = outputs[0].size(1) - 1
            l_start = torch.zeros(outputs[0].size(1))
            l_end = torch.zeros(outputs[0].size(1))
            num_nodes = c.num_nodes()
            mini = torch.min(tok).item()
            maxi = torch.max(tok).item()
            e_ind = min(num_nodes - mini, lm)
            s_ind = min(num_nodes - maxi, lm)
            if s_ind == start_positions.item() or e_ind == end_positions.item():
                l_start[s_ind] = 5
                l_end[e_ind] = 5

        data = transform(data)

        data.to("cuda:0")

        tokens = (
            graph_data[0]["graph"]
            .to_dict()["token"]["x"]
            .squeeze()[
                torch.nonzero(graph_data[0]["graph"].to_dict()["token"]["x"].squeeze())
            ]
            .squeeze()
        )
        positions = (
            graph_data[0]["graph"].to_dict()["constituent"]["y"].float().to("cuda:0")
        )

        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        for node_type, x in x_dict.items():

            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        logits = self.graph_qa_outputs(x_dict["constituent"])
        logits2 = self.graph2_qa_outputs(x_dict["token"])
        start_logits, end_logits = logits2.split(1, dim=-1)
        logits.to("cuda:0")
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        logits = logits.squeeze(1)
        l_start = torch.zeros(outputs[0].size(1))
        l_end = torch.zeros(outputs[0].size(1))

        kop = kop.to("cuda:0")
        y = y.to("cuda:0")

        m = torch.argmax(logits).item()
        if m != 0:
            result = torch.cat(dgl.traversal.bfs_nodes_generator(c, m))
            msk = result.ge(g.num_nodes("constituent"))
            tok = torch.masked_select(result, msk)
            lm = outputs[0].size(1) - 1
            e_index = min(c.num_nodes() - torch.min(tok).item(), lm)
            s_index = min(c.num_nodes() - torch.max(tok).item(), lm)
            l_start[s_index] = torch.max(logits).item()
            l_end[e_index] = torch.max(logits).item()

        l_end = l_end.to("cuda:0")
        l_start = l_start.to("cuda:0")
        start_logits = start_logits.to("cuda:0")
        end_logits = end_logits.to("cuda:0")
        s_logits = start_logits.add(l_start)
        e_logits = end_logits.add(l_end)
        y = y.to("cuda:0")
        total_loss = None

        start_positions = start_positions.to("cuda:0")
        end_positions = end_positions.to("cuda:0")

        if start_positions is not None and end_positions is not None and test == False:
            loss_fct = nn.CrossEntropyLoss()
            s_logits = s_logits.unsqueeze(0)
            e_logits = e_logits.unsqueeze(0)
            start_loss = loss_fct(s_logits, start_positions)
            end_loss = loss_fct(e_logits, end_positions)
            loss = loss_fct(logits, y)
            total_loss = (start_loss + end_loss + loss) / 3

        if not return_dict:
            output = (s_logits, e_logits) + outputs[2:]
            return (
                ((total_loss,) + output)
                if total_loss is not None
                else [output, lp_type]
            )

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=s_logits,
            end_logits=e_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    print("made it")
