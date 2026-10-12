# SPDX-License-Identifier: AGPL-3.0-only
"""Keep calibrated A16W8 parameters; move sensitive output math to CPU.

Candidates are separate models and are never selected before hardware validation.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import onnx
import numpy as np
from onnx import helper, numpy_helper
from app.model_utils import MODEL, ROOT, sha256, write_json, cases, decode, compare_detections, output_pair
from scripts.prepare_models import QUANT, cpu


def transform(model, variant):
    m = copy.deepcopy(model)
    if variant in ('cpu-score-outside','cpu-classifier-outside','cpu-head-outside','cpu-head-silu-outside','cpu-head-conv-outside'):
        producers={v:n for n in m.graph.node for v in n.output}
        sigmoid=next(n for n in m.graph.node if n.name=='/model/model.23/Sigmoid')
        outputs=[m.graph.output[0].name]
        if variant != 'cpu-score-outside':
            for i,s in enumerate((8,16,32)):
                suffix=f'/one2one_cv3.{i}.2/Conv' if variant=='cpu-classifier-outside' else f'/one2one_cv3.{i}.0.0/conv/Conv'
                node=next(n for n in m.graph.node if n.op_type=='Conv' and n.name.endswith(suffix))
                feature=node.input[0]
                wnode=producers[node.input[1]]
                initials={v.name:numpy_helper.to_array(v) for v in m.graph.initializer}
                group=next((helper.get_attribute_value(a) for a in node.attribute if a.name=='group'),1)
                channels=initials[wnode.input[0]].shape[1]*group
                if variant in ('cpu-head-silu-outside','cpu-head-conv-outside'):
                    dqnode=producers[feature]
                    qnode=producers[dqnode.input[0]]
                    mul=producers[qnode.input[0]]
                    if mul.op_type!='Mul':raise RuntimeError('Expected backbone SiLU')
                    feature=mul.input[0]
                    if variant=='cpu-head-conv-outside':
                        d=producers[feature];q=producers[d.input[0]];conv=producers[q.input[0]]
                        if conv.op_type!='Conv':raise RuntimeError('Expected backbone final Conv')
                        feature=conv.input[0]
                        weights=producers[conv.input[1]]
                        group=next((helper.get_attribute_value(a) for a in conv.attribute if a.name=='group'),1)
                        channels=initials[weights.input[0]].shape[1]*group
                if not any(v.name==feature for v in m.graph.value_info):
                    m.graph.value_info.append(helper.make_tensor_value_info(feature,onnx.TensorProto.FLOAT,[1,channels,448//s,448//s]))
                outputs.append(feature)
        else:
            logits=sigmoid.input[0]
            if not any(v.name==logits for v in m.graph.value_info):
                m.graph.value_info.append(helper.make_tensor_value_info(logits,onnx.TensorProto.FLOAT,[1,80,4116]))
            outputs.append(logits)
        extracted=onnx.utils.Extractor(m).extract_model(['images'],outputs)
        onnx.checker.check_model(extracted)
        return extracted,[sigmoid.name]
    nodes = []
    one = 'demo_one'
    m.graph.initializer.append(numpy_helper.from_array(np.array(1., np.float32), one))
    axes = 'demo_squeeze_axes'
    m.graph.initializer.append(numpy_helper.from_array(np.array([2,3], np.int64), axes))
    biasaxes = 'demo_bias_axes'
    m.graph.initializer.append(numpy_helper.from_array(np.array([0,2,3], np.int64), biasaxes))
    replaced = []
    for n in m.graph.node:
        if n.name == '/model/model.23/Sigmoid':
            # Float Exp/Reciprocal without intermediate QDQ is placed on CPU.
            t = n.name+'_demo'
            nodes += [helper.make_node('Neg', [n.input[0]], [t+'_neg'], name=t+'_neg'),
                      helper.make_node('Exp', [t+'_neg'], [t+'_exp'], name=t+'_exp'),
                      helper.make_node('Add', [t+'_exp', one], [t+'_den'], name=t+'_den'),
                      helper.make_node('Reciprocal', [t+'_den'], list(n.output), name=t+'_reciprocal')]
            replaced.append(n.name)
        elif variant == 'cpu-classifier' and n.op_type == 'Conv' and any(
                n.name.endswith(f'/one2one_cv3.{i}.2/Conv') for i in range(3)):
            attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
            if attrs.get('kernel_shape') != [1,1] or attrs.get('group',1) != 1:
                raise RuntimeError('Expected 1x1 output classifier')
            t = n.name+'_demo'
            # DQ weight/bias/input remain unchanged (A16W8). Einsum uses CPU.
            # Opset17 supports Squeeze/Unsqueeze axes as the second input.
            nodes += [helper.make_node('Squeeze',[n.input[1],axes],[t+'_weight'],name=t+'_weight'),
                      helper.make_node('Einsum',[n.input[0],t+'_weight'],[t+'_product'],name=t+'_einsum',equation='bchw,oc->bohw'),
                      helper.make_node('Unsqueeze',[n.input[2],biasaxes],[t+'_bias'],name=t+'_bias'),
                      helper.make_node('Add',[t+'_product',t+'_bias'],list(n.output),name=t+'_add')]
            replaced.append(n.name)
        else:
            nodes.append(n)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.checker.check_model(m)
    return m, replaced


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('variant', choices=['cpu-sigmoid','cpu-classifier','cpu-score-outside','cpu-classifier-outside','cpu-head-outside','cpu-head-silu-outside','cpu-head-conv-outside'])
    args = p.parse_args()
    folder = ROOT/'models/npu/candidates'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/f'yolo26n-448-{args.variant}-int16.onnx'
    reference = folder/f'yolo26n-448-{args.variant}.onnx'
    if target.exists() or reference.exists():
        raise SystemExit('Candidate already exists; move aside to regenerate')
    base_meta = json.loads(MODEL.with_suffix('.json').read_text())
    if args.variant in ('cpu-score-outside','cpu-classifier-outside'):
        original=onnx.load(QUANT)
        initializers={v.name:numpy_helper.to_array(v) for v in original.graph.initializer}
        q=next(n for n in original.graph.node if n.name=='/model/model.23/Sigmoid_output_0_QuantizeLinear')
        base_meta['score_transform']=dict(scale=float(initializers[q.input[1]].item()),zero_point=int(initializers[q.input[2]].item()))
    if args.variant=='cpu-classifier-outside':
        nodes={v:n for n in original.graph.node for v in n.output}
        def dq(name):
            n=nodes[name];values,scale,zero=[initializers[v] for v in n.input]
            attrs={a.name:helper.get_attribute_value(a) for a in n.attribute}
            if scale.size>1:
                shape=[1]*values.ndim;shape[attrs.get('axis',1)]=scale.size
                scale,zero=scale.reshape(shape),zero.reshape(shape)
            return (values.astype(np.float32)-zero)*scale
        def qp(prefix):
            node=next(n for n in original.graph.node if n.name==prefix+'_QuantizeLinear')
            return dict(scale=float(initializers[node.input[1]].item()),zero_point=int(initializers[node.input[2]].item()))
        base_meta['cpu_classifiers']=[]
        for i in range(3):
            n=next(n for n in original.graph.node if n.op_type=='Conv' and n.name.endswith(f'/one2one_cv3.{i}.2/Conv'))
            base_meta['cpu_classifiers'].append(dict(weights=dq(n.input[1])[:,:,0,0].tolist(),bias=dq(n.input[2]).tolist(),
                qdq_stages=[qp(n.output[0]),qp(f'/model/model.23/Reshape_{3+i}_output_0')]))
        base_meta['concat_qdq']=qp('/model/model.23/Concat_1_output_0')
    m, replaced = transform(onnx.load(MODEL),args.variant)
    r, _ = transform(onnx.load(QUANT),args.variant)
    old, new = cpu(QUANT), cpu(r.SerializeToString())
    head_session=None
    headpath=folder/f'yolo26n-448-{args.variant}-head.onnx'
    if args.variant in ('cpu-head-outside','cpu-head-silu-outside','cpu-head-conv-outside'):
        original=onnx.load(QUANT)
        features=[v.name for v in r.graph.output[1:]]
        for v in r.graph.output[1:]:
            if not any(x.name==v.name for x in original.graph.value_info):
                original.graph.value_info.append(copy.deepcopy(v))
        head=onnx.utils.Extractor(original).extract_model(features,[original.graph.output[1].name])
        onnx.checker.check_model(head)
        head_session=cpu(head.SerializeToString())
    checks = []
    for record,x in cases():
        a,b = old.run(None,{'images':x}),new.run(None,{'images':x})
        pair=(b[0],head_session.run(None,dict(zip(features,b[1:])))[0]) if head_session else output_pair(b,base_meta)
        check=compare_detections(decode(*a),decode(*pair))
        checks.append(dict(input=record, raw_max_abs=[float(np.max(np.abs(u-v))) for u,v in zip(a,pair)], **check))
        if not check['passed']:
            raise RuntimeError('Candidate CPU behavior changed beyond the smoke criterion')
    onnx.save(r,reference)
    onnx.save(m,target)
    if head_session:
        onnx.save(head,headpath)
        base_meta.update(cpu_head_model=str(headpath),cpu_head_sha256=sha256(headpath),cpu_head_input_names=features)
        replaced=[n.name for n in head.graph.node if n.op_type not in ('QuantizeLinear','DequantizeLinear','Constant')]
    shapes=[[d.dim_value for d in v.type.tensor_type.shape.dim] for v in m.graph.output]
    write_json(target.with_suffix('.json'),dict(base_meta,model_sha256=sha256(target),output_shapes=shapes,
        reference_model=str(QUANT if head_session else reference),reference_sha256=sha256(QUANT if head_session else reference),variant=args.variant,
        parent_model_sha256=sha256(MODEL),cpu_math_nodes=replaced,cpu_checks=checks,
        note='All calibrated QDQ parameters preserved; sensitive output math moved to CPU'))
    print(target)
