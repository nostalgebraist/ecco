import torch
import transformers
import ecco
import ecco.torch_util
from torch.nn import functional as F
import numpy as np
import pandas as pd
from ecco.output import OutputSeq
import random
from IPython import display as d
import os
import json
from ecco.attribution import *
from typing import Optional, Any
from transformers.modeling_gpt2 import GPT2Model
from tqdm.autonotebook import tqdm, trange


# TODO: move
def detupleize_gram(g):
  return "|".join([str(t) for t in g])

def retupleize_gram(s):
  return tuple([int(t) for t in s.split("|")])

def sample_output_token(scores, do_sample, temperature, top_k, top_p):
    if do_sample:
        # Temperature (higher temperature => more likely to sample low probability tokens)
        if temperature != 1.0:
            scores = scores / temperature
        # Top-p/top-k filtering
        next_token_logscores = transformers.generation_utils. \
            top_k_top_p_filtering(scores,
                                  top_k=top_k,
                                  top_p=top_p)
        # Sample
        probs = F.softmax(next_token_logscores, dim=-1)
        # print(probs.shape)
        prediction_id = torch.multinomial(probs, num_samples=1)
    else:
        # Greedy decoding
        prediction_id = torch.argmax(scores, dim=-1)
    prediction_id = prediction_id.squeeze()
    return prediction_id


def _one_hot(token_ids, vocab_size):
    return torch.zeros(len(token_ids), vocab_size).scatter_(1, token_ids.unsqueeze(1), 1.)


def activations_dict_to_array(activations_dict):
    # print(activations_dict[0].shape)
    activations = []
    for i in sorted(activations_dict.keys()):
        activations.append(activations_dict[i])

    activations = np.concatenate(activations, axis=0)
    return np.swapaxes(activations, 1, 2)


class LM(object):
    """
    Wrapper around language model. Provides saliency for generated tokens and collects neuron activations.
    """

    def __init__(self, model, tokenizer,
                 collect_activations_flag=False,
                 collect_gen_activations_flag=False,
                 collect_activations_layer_nums=None,  # None --> collect for all layers
                 ):
        self.model = model
        if torch.cuda.is_available():
            self.model = model.to('cuda')

        self.tokenizer = tokenizer
        self._path = os.path.dirname(ecco.__file__)

        self.device = 'cuda' if torch.cuda.is_available() and self.model.device.type == 'cuda' \
            else 'cpu'

        # Neuron Activation
        self.collect_activations_flag = collect_activations_flag
        self.collect_gen_activations_flag = collect_gen_activations_flag
        self.collect_activations_layer_nums = collect_activations_layer_nums
        self._hooks = {}
        self._reset()
        self._attach_hooks(self.model)

        # If running in Jupyer, outputting setup this in one cell is enough. But for colab
        # we're running it before every d.HTML cell
        # d.display(d.HTML(filename=os.path.join(self._path, "html", "setup.html")))

    def _reset(self):
        self._all_activations_dict = {}
        self._generation_activations_dict = {}
        self._attn_out_dict = {}  # TODO: better name
        self._mlp_out_dict = {}  # TODO: better name
        self.activations = []
        self.all_activations = []
        self.generation_activations = []
        self.neurons_to_inhibit = {}
        self.neurons_to_induce = {}

    def to(self, tensor: torch.Tensor):
        if self.device == 'cuda':
            return tensor.to('cuda')
        return tensor

    def _generate_token(self, input_ids, past, do_sample: bool, temperature: float, top_k: int, top_p: float,
                        attribution_flag: Optional[bool]):
        """
        Run a forward pass through the model and sample a token.
        """
        inputs_embeds, token_ids_tensor_one_hot = self._get_embeddings(input_ids)

        output = self.model(inputs_embeds=inputs_embeds, return_dict=True, use_cache=False)
        predict = output.logits

        scores = predict[-1:, :]

        prediction_id = sample_output_token(scores, do_sample, temperature, top_k, top_p)
        # Print the sampled token
        # print(self.tokenizer.decode([prediction_id]))

        # prediction_id now has the id of the token we want to output
        # To do feature importance, let's get the actual logit associated with
        # this token
        prediction_logit = predict[inputs_embeds.shape[0] - 1][prediction_id]

        if attribution_flag:
            saliency_results = compute_saliency_scores(prediction_logit, token_ids_tensor_one_hot, inputs_embeds)

            if 'gradient' not in self.attributions:
                self.attributions['gradient'] = []
            self.attributions['gradient'].append(saliency_results['gradient'].cpu().detach().numpy())

            if 'grad_x_input' not in self.attributions:
                self.attributions['grad_x_input'] = []
            self.attributions['grad_x_input'].append(saliency_results['grad_x_input'].cpu().detach().numpy())

        output['logits'] = None  # free tensor memory we won't use again

        # detach(): don't need grads here
        # cpu(): not used by GPU during generation; may lead to GPU OOM if left on GPU during long generations
        if getattr(output, "hidden_states", None) is not None:
            output.hidden_states = tuple([h.cpu().detach() for h in output.hidden_states])

        return prediction_id, output

    def generate(self, input_str: str, max_length: Optional[int] = 128,
                 temperature: Optional[float] = None,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 get_model_output: Optional[bool] = False,
                 do_sample: Optional[bool] = None,
                 attribution: Optional[bool] = True,
                 generate: Optional[int] = None):

        top_k = top_k if top_k is not None else self.model.config.top_k
        top_p = top_p if top_p is not None else self.model.config.top_p
        temperature = temperature if temperature is not None else self.model.config.temperature
        do_sample = do_sample if do_sample is not None else self.model.config.task_specific_params['text-generation'][
            'do_sample']

        input_ids = self.tokenizer(input_str, return_tensors="pt")['input_ids'][0]
        n_input_tokens = len(input_ids)

        if generate is not None:
            max_length = n_input_tokens + generate

        past = None
        self.attributions = {}
        outputs = []

        cur_len = len(input_ids)

        if cur_len >= max_length:
            raise (ValueError,
                   "max_length set to {} while input token has more tokens ({}). Consider increasing max_length" \
                   .format(max_length, cur_len))

        # Print output
        viz_id = self.display_input_sequence(input_ids)

        while cur_len < max_length:
            output_token_id, output = self._generate_token(input_ids,
                                                           past,
                                                           # Note, this is not currently used
                                                           temperature=temperature,
                                                           top_k=top_k, top_p=top_p,
                                                           do_sample=do_sample,
                                                           attribution_flag=attribution)

            if (get_model_output):
                outputs.append(output)
            input_ids = torch.cat([input_ids, torch.tensor([output_token_id])])

            self.display_token(viz_id,
                               output_token_id.cpu().numpy(),
                               cur_len)
            cur_len = cur_len + 1

            if output_token_id == self.model.config.eos_token_id:
                break

        # Turn activations from dict to a proper array
        activations_dict = self._all_activations_dict or self._generation_activations_dict

        if activations_dict != {}:
            self.activations = activations_dict_to_array(activations_dict)

        hidden_states = getattr(output, "hidden_states", None)
        tokens = []
        for i in input_ids:
            token = self.tokenizer.decode([i])
            tokens.append(token)

        attributions = self.attributions
        attn = getattr(output, "attentions", None)

        # TODO: cleanup calculation
        attn_outs = [self._attn_out_dict[k][0] for k in sorted(self._attn_out_dict.keys())]
        mlp_outs = [self._mlp_out_dict[k][0] for k in sorted(self._mlp_out_dict.keys())]

        return OutputSeq(**{'tokenizer': self.tokenizer,
                            'token_ids': input_ids,
                            'n_input_tokens': n_input_tokens,
                            'output_text': self.tokenizer.decode(input_ids),
                            'tokens': tokens,
                            'hidden_states': hidden_states,
                            'attention': attn,
                            'model_outputs': outputs,
                            'attribution': attributions,
                            'activations': self.activations,
                            'collect_activations_layer_nums': self.collect_activations_layer_nums,
                            'lm_head': self.model.lm_head,
                            'attn_outs': attn_outs,
                            'mlp_outs': mlp_outs,
                            'layer_norm': self.layer_norm,
                            'layer_norm_f': self.layer_norm_f,
                            'device': self.device})

    def _get_embeddings(self, input_ids):
        """
        Takes the token ids of a sequence, returnsa matrix of their embeddings.
        """
        embedding_matrix = self.model.transformer.wte.weight

        vocab_size = embedding_matrix.shape[0]
        one_hot_tensor = self.to(_one_hot(input_ids, vocab_size))

        token_ids_tensor_one_hot = one_hot_tensor.clone().requires_grad_(True)
        # token_ids_tensor_one_hot.requires_grad_(True)

        inputs_embeds = torch.matmul(token_ids_tensor_one_hot, embedding_matrix)
        return inputs_embeds, token_ids_tensor_one_hot

    def _attach_hooks(self, model):
        for name, module in model.named_modules():
            # Add hooks to capture activations in every FFNN
            if "mlp.c_proj" in name:
                if self.collect_activations_flag:
                    self._hooks[name] = module.register_forward_hook(
                        lambda self_, input_, output,
                               name=name: self._get_activations_hook(name, input_))

                if self.collect_gen_activations_flag:
                    self._hooks[name] = module.register_forward_hook(
                        lambda self_, input_, output,
                               name=name: self._get_generation_activations_hook(name, input_))

                # Register neuron inhibition hook
                self._hooks[name + '_inhibit'] = module.register_forward_pre_hook(
                    lambda self_, input_, name=name: \
                        self._inhibit_neurons_hook(name, input_)
                )
        self._attach_subblock_hooks(model)

    def _attach_subblock_hooks(self, model):
        for name, module in model.named_modules():
            if "attn.c_proj" in name:
                self._hooks[name] = module.register_forward_hook(
                    lambda self_, input_, output,
                            name=name: self._get_attn_out_hook(name, output))

            if "mlp.c_proj" in name:
                self._hooks[name] = module.register_forward_hook(
                    lambda self_, input_, output,
                            name=name: self._get_mlp_out_hook(name, output))

    def _get_attn_out_hook(self, name, out):
      layer_number = int(name.split('.')[2])

      if layer_number not in self._attn_out_dict:
          self._attn_out_dict[layer_number] = [0]

      self._attn_out_dict[layer_number][0] = (out[0].detach().cpu())

    def _get_mlp_out_hook(self, name, out):
      layer_number = int(name.split('.')[2])

      if layer_number not in self._mlp_out_dict:
          self._mlp_out_dict[layer_number] = [0]

      self._mlp_out_dict[layer_number][0] = (out[0].detach().cpu())

    def _get_activations_hook(self, name: str, input_):
        """
        Collects the activation for all tokens (input and output)
        """
        # print(input_.shape, output.shape)
        # in distilGPT and GPT2, the layer name is 'transformer.h.0.mlp.c_fc'
        # Extract the number of the layer from the name
        layer_number = int(name.split('.')[2])

        collecting_this_layer = (self.collect_activations_layer_nums is None) or (layer_number in self.collect_activations_layer_nums)

        if collecting_this_layer:
            if layer_number not in self._all_activations_dict:
                self._all_activations_dict[layer_number] = [0]

            # Overwrite the previous step activations. This collects all activations in the last step
            # Assuming all input tokens are presented as input, no "past"
            # The inputs to c_proj already pass through the gelu activation function
            self._all_activations_dict[layer_number][0] = input_[0][0].detach().cpu().numpy()

    def _get_generation_activations_hook(self, name: str, input_):
        """
        Collects the activation for the token being generated
        """
        # print(input_.shape, output.shape)
        # in distilGPT and GPT2, the layer name is 'transformer.h.0.mlp.c_fc'
        # Extract the number of the layer from the name
        layer_number = int(name.split('.')[2])

        collecting_this_layer = (self.collect_activations_layer_nums is None) or (layer_number in self.collect_activations_layer_nums)

        if collecting_this_layer:
            if layer_number not in self._generation_activations_dict:
                self._generation_activations_dict[layer_number] = []

            # Accumulate in dict
            # The inputs to c_proj already pass through the gelu activation function
            self._generation_activations_dict[layer_number].append(input_[0][0][-1].detach().cpu().numpy())

    def _inhibit_neurons_hook(self, name: str, input_tensor):
        """
        After being attached as a pre-forward hook, it sets to zero the activation value
        of the neurons indicated in self.neurons_to_inhibit
        """

        layer_number = int(name.split('.')[2])
        if layer_number in self.neurons_to_inhibit.keys():
            # print('layer_number', layer_number, input_tensor[0].shape)

            for n in self.neurons_to_inhibit[layer_number]:
                # print('inhibiting', layer_number, n)
                input_tensor[0][0][-1][n] = 0  # tuple, batch, position

        if layer_number in self.neurons_to_induce.keys():
            # print('layer_number', layer_number, input_tensor[0].shape)

            for n in self.neurons_to_induce[layer_number]:
                # print('inhibiting', layer_number, n)
                input_tensor[0][0][-1][n] = input_tensor[0][0][-1][n] * 10  # tuple, batch, position

        return input_tensor

    @property
    def layer_norm(self):
        ln_no_affine = torch.nn.LayerNorm((self.model.transformer.ln_f.weight.shape[0],), elementwise_affine=False)
        self.to(ln_no_affine)
        return ln_no_affine

    @property
    def layer_norm_f(self):
        return self.model.transformer.ln_f

    def get_ngram_mlp_activations(self, indices, layer_num: int, batchsize_base=128, nbatch=None):
        n_indices = len(indices)
        gram_size = indices.shape[1]
        gram_range = list(range(gram_size))

        batchsize = batchsize_base // gram_size

        if nbatch is None:
            nbatch = n_indices//batchsize + 1

        results = []

        for i in trange(0, min(n_indices, nbatch*batchsize), batchsize):
            batch_raw_input_ids = indices[i:i+batchsize, :]
            batch_input_ids = np.concatenate([row for row in batch_raw_input_ids])
            batch_position_ids = np.concatenate([gram_range for row in batch_raw_input_ids])

            _ = self.model.forward(input_ids=ecco.torch_util.to_tensor(batch_input_ids, self.device),
                                    position_ids=ecco.torch_util.to_tensor(batch_position_ids, self.device))

            acts_raw = self._all_activations_dict[layer_num][0]
            acts = np.stack([acts_raw[:, j].reshape(-1, gram_size)[:, -1] for j in range(acts_raw.shape[1])], axis=1)
            results.append(acts)

        results_all = np.concatenate(results, axis=0)
        df = pd.DataFrame(results_all.T, columns=[detupleize_gram(row) for row in indices])
        return df

    def get_token_mlp_activations(self, layer_num: int, batchsize=128, nbatch=None, indices=None):
        n_vocab=self.tokenizer.vocab_size

        if indices is None:
            indices = np.arange(0, n_vocab)
        indices = indices.reshape(-1, 1)

        return self.get_ngram_mlp_activations(
            indices=indices,
            layer_num=layer_num,
            batchsize_base=batchsize,
            nbatch=nbatch,
        )


    def visualize_token_activations(self, token_activations, i,
                                    max_tokens_to_show=100,
                                    max_tokens_to_show_mid=20,
                                    max_tokens_to_show_neg=20,
                                    cutoff_pos=1,
                                    cutoff_neg=-0.1,
                                    do_gelu=False,
                                    use_cutoffs=True):
        def gelu(x):
            return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3)))

        token_activations_df = token_activations
        token_activations = token_activations_df.values

        this_token_activations = token_activations[i, :]
        if use_cutoffs and (min(this_token_activations) > cutoff_pos):
              return "Not selective enough"

        ixs = this_token_activations.argsort()
        pos_ixs = ixs[this_token_activations[ixs]>=0]
        neg_ixs = ixs[this_token_activations[ixs]<0]

        top_ixs = pos_ixs[-max_tokens_to_show:][::-1]
        print(f"{len(top_ixs)} top")

        top_ixs_mid = pos_ixs[:max_tokens_to_show_mid][::-1]
        top_ixs_mid = [ix for ix in top_ixs_mid if ix not in top_ixs]
        print(f"{len(top_ixs_mid)} mid")
        top_ixs = np.concatenate([top_ixs, top_ixs_mid]).astype(int)

        top_ixs_neg = neg_ixs[:max_tokens_to_show_neg][::-1]
        top_ixs_neg = [ix for ix in top_ixs_neg if ix not in top_ixs]
        print(f"{len(top_ixs_neg)} neg")
        top_ixs = np.concatenate([top_ixs, top_ixs_neg]).astype(int)

        token_ids = []
        tokens = []
        activn_ins = []
        activns = []

        _activn_ins = this_token_activations[top_ixs]
        if do_gelu:
            _activns = gelu(_activn_ins)
        else:
            _activns = _activn_ins

        activn_in_max = _activns.max()

        for token_ix, _activn in zip(top_ixs, _activns):
              token_id = token_activations_df.columns[token_ix]
              activn_in = this_token_activations[token_ix]

              if (not use_cutoffs) or ((_activn > cutoff_pos) or (_activn < cutoff_neg)):
                token_ids.append(token_id)
                tokens.append(" " +
                    self.tokenizer.convert_tokens_to_string(
                        self.tokenizer.convert_ids_to_tokens([token_id])
                        )
                )
                activn_ins.append(activn_in)
                activns.append(_activn)

        if len(token_ids)==0:
              return "Not selective enough (no tokens to show)"
        activn_ins = np.asarray(activn_ins)
        activns = np.asarray(activns)
        print(activns.shape)

        factors = [np.stack([
                                  np.clip(activns, a_min=0, a_max=None),
                                  -1*np.clip(activns, a_min=None, a_max=0),
                                  ])]
        # print((activn_ins.min(), activn_ins.max()))
        # print((factors[0][0, 0], factors[0][0, -1]))

        fake_output = ecco.output.OutputSeq(
                token_ids=token_ids,
                tokens=tokens,
                n_input_tokens=0,
                )
        ecco.lm_plots.explore_arbitrary_sparklines(fake_output,
                                        factors
                                        )

    def visualize_multiple_token_activations(self, token_activations, indices,
                                              indices_to_names={},
                                              reference_activations=None,
                                              max_tokens_to_show=100,
                                              max_tokens_to_show_mid=20,
                                              max_tokens_to_show_neg=20,
                                              cutoff_pos=1,
                                              cutoff_neg=-0.1,
                                              do_gelu=False,
                                              use_cutoffs=True):
        for i in indices:
            namef = f" ({indices_to_names.get(i, '')})" if i in indices_to_names else ""
            msg = f"spike {i}{namef}"
            if reference_activations is not None:
                msg += f"\nACT: {reference_activations[i]:.2f}"
            msg += "\n"
            print(msg)

            retval=self.visualize_token_activations(
                token_activations,
                i,
                max_tokens_to_show=max_tokens_to_show,
                max_tokens_to_show_mid=max_tokens_to_show_mid,
                max_tokens_to_show_neg=max_tokens_to_show_neg,
                cutoff_pos=cutoff_pos,
                cutoff_neg=cutoff_neg,
                do_gelu=do_gelu,
                use_cutoffs=use_cutoffs,
                )

            if retval is None:
              print('\n\n\n\n----------\n\n\n\n\n\n\n\n\n')
            else:
              print(retval + "\n")

    def display_input_sequence(self, input_ids):

        tokens = []
        for idx, token_id in enumerate(input_ids):
            type = "input"
            tokens.append({'token': self.tokenizer.decode([token_id]),
                           'position': idx,
                           'token_id': int(token_id),
                           'type': type})
        data = {'tokens': tokens}

        d.display(d.HTML(filename=os.path.join(self._path, "html", "setup.html")))
        d.display(d.HTML(filename=os.path.join(self._path, "html", "basic.html")))
        viz_id = f'viz_{round(random.random() * 1000000)}'
#         html = f"""
# <div id='{viz_id}_output'></div>
# <script>
# """

        js = f"""

         requirejs( ['basic', 'ecco'], function(basic, ecco){{
            basic.init('{viz_id}')

            window.ecco['{viz_id}'] = ecco.renderOutputSequence('{viz_id}', {data})
         }}, function (err) {{
            console.log(err);
        }})
"""
        # print(js)
        # d.display(d.HTML(html))
        d.display(d.Javascript(js))
        return viz_id

    def display_token(self, viz_id, token_id, position):
        token = {
            'token': self.tokenizer.decode([token_id]),
            'token_id': int(token_id),
            'position': position,
            'type': 'output'
        }
        js = f"""
        // We don't really need these require scripts. But this is to avert
        //this code from running before display_input_sequence which DOES require external files
        requirejs(['basic', 'ecco'], function(basic, ecco){{
                console.log('addToken viz_id', '{viz_id}');
                window.ecco['{viz_id}'].addToken({json.dumps(token)})
                window.ecco['{viz_id}'].redraw()
        }})
        """
        # print(js)
        d.display(d.Javascript(js))

    def predict_token(self, inputs, topk=50, temperature=1.0):

        output = self.model(**inputs)
        scores = output[0][0][-1] / temperature
        s = scores.detach().numpy()
        sorted_predictions = s.argsort()[::-1]
        sm = F.softmax(scores, dim=-1).detach().numpy()

        tokens = [self.tokenizer.decode([t]) for t in sorted_predictions[:topk]]
        probs = sm[sorted_predictions[:topk]]

        prediction_data = []
        for idx, (token, prob) in enumerate(zip(tokens, probs)):
            # print(idx, token, prob)
            prediction_data.append({'token': token,
                                    'prob': str(prob),
                                    'ranking': idx + 1,
                                    'token_id': str(sorted_predictions[idx])
                                    })

        params = prediction_data

        viz_id = 'viz_{}'.format(round(random.random() * 1000000))

        d.display(d.HTML(filename=os.path.join(self._path, "html", "predict_token.html")))
        js = """
        requirejs(['predict_token'], function(predict_token){{
        if (window.predict === undefined)
            window.predict = {{}}
        window.predict["{}"] = new predict_token.predictToken("{}", {})
        }}
        )
        """.format(viz_id, viz_id, json.dumps(params))
        d.display(d.Javascript(js))


class MockGPT(GPT2Model):
    def __init__(self):
        print('Mock tokenizer init')
        config = transformers.GPT2Config.from_pretrained("gpt2")
        super().__init__(config)
        self.transformer ={'wte':{'weight':  torch.Tensor([])}}

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __call__(self, **kwargs):
        print('calling model', kwargs)
        return OutputSeq(**{'tokenizer': MockGPTTokenizer(),
                            'token_ids': [352, 11, 352, 11, 362],
                            'n_input_tokens': 4,
                            'output_text': ' 1, 1, 2',
                            'tokens': [' 1', ',', ' 1', ',', ' 2'],
                            'hidden_states': [torch.rand(4, 768) for i in range(7)],
                            'attention': None,
                            'model_outputs': None,
                            'attribution': {'gradient': [
                                np.array([0.41861308, 0.13054065, 0.23851791, 0.21232839], dtype=np.float32)],
                                'grad_x_input': [
                                    np.array([0.31678662, 0.18056837, 0.37555906, 0.12708597],
                                             dtype=np.float32)]},
                            'activations': [],
                            'lm_head': torch.nn.Linear(768, 50257, bias=False),
                            'device': 'cpu'})

    # def generate(self, input_str, generate=1, **kwargs):


class MockGPTTokenizer(transformers.PreTrainedTokenizer):
    def __init__(self):
        super().__init__()

        print('Mock model init')
