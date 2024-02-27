import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
from fiftyone.brain import Similarity
from pprint import pprint
from fiftyone import ViewField as F


import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
import fiftyone.core.utils as fou
import numpy as np
import fiftyone.zoo as foz

import scipy




class OptConfThresh(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="optimal_conf_threshold",
            label="Find Your Optimal Confidence Threshold",
            description="Finds optimal confidece threshold based on F1 score for your detection datasets",
            icon="/assets/binoculars.svg",
            dynamic=True,

        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
    
        ready = _opt_conf_thresh_inputs(ctx,inputs)

        if ready:
            _execution_mode(ctx, inputs)
        

        return types.Property(inputs)
    
    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)
    
    def execute(self, ctx):
        
        f1score, conf_thresh = _opt_conf_thresh(ctx)
    
        return {"f1score": str(f1score), "conf_thresh": str(conf_thresh)}
    
    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("f1score", label="Best F1 Score")
        outputs.str("conf_thresh", label="Best conf_thresh")
        header = "Optimal Confidence Theshold"
    
        return types.Property(outputs, view=types.View(label=header))
    
def _opt_conf_thresh_inputs(ctx, inputs):

    target_view = get_target_view(ctx, inputs)

    inputs.float(
            "lower_bound",
            label="Lower Bound",
            description="What is the lower bound of the thresholds you would like to check? Choose a number bounded (0,1)",
            view=types.FieldView(componentsProps={'field': {'min': 0.0001, "max": 0.9999, "step": 0.01, "default": 0.01}}),
            )
    
    inputs.float(
            "upper_bound",
            label="Upper Bound",
            description="What is the upper bound of the thresholds you would like to check? Choose a number bounded (0,1)",
            view=types.FieldView(componentsProps={'field': {'min': 0.0001, "max": 0.9999, "step": 0.01, "default": 0.01}}),
            )

    
    labels = []
    field_names = list(target_view.get_field_schema().keys())
    for name in field_names:
        if type(target_view.get_field(name)) == fo.core.fields.EmbeddedDocumentField:
            if "detections" in  list(target_view.get_field(name).get_field_schema().keys()):
                labels.append(name)

    if labels == []:
        inputs.view(
        "error", 
        types.Error(label="No labels found on this dataset", description="Add labels to be able to filter by them")
    )
    else:

        label_radio_group = types.RadioGroup()

        for choice in labels:
            label_radio_group.add_choice(choice, label=choice)

        inputs.enum(
            "label_ground_truth",
            label_radio_group.values(),
            label="Choose Ground Truth",
            description="Choose what label field is the ground_truth:",
            view=types.DropdownView(),
            required=True,
            default=None
            )
        inputs.enum(
            "label_predictions",
            label_radio_group.values(),
            label="Choose Predictions",
            description="Choose what label field are the predictions:",
            view=types.DropdownView(),
            required=True,
            default=None
            )
        
        gt =   ctx.params.get("label_ground_truth")
        pred =  ctx.params.get("label_predictions")
        if gt == pred:
            inputs.view(
                "error", 
                types.Error(label="Fields Cannot Be The Same!", description="Select two different label fields to begin!")
            )


    return True



def _opt_conf_thresh(ctx):
    lb = ctx.params.get("lower_bound")
    ub = ctx.params.get("upper_bound")
   
    best_f1 = -1
    best_threshold = None
    gt =   ctx.params.get("label_ground_truth")
    pred =  ctx.params.get("label_predictions")
    res = scipy.optimize.fminbound(
                    func=calculate_f1,
                    x1=lb,
                    x2=ub,
                    args=(ctx,pred,gt),
                    xtol=0.01,
                    full_output=True
    )

    best_conf, f1val, ierr, numfunc = res
    return -1.0*f1val, best_conf


def calculate_f1(conf, ctx, pred, gt):
    conf_view = ctx.dataset.filter_labels(pred, F("confidence") >= conf)
    results = conf_view.evaluate_detections(pred,
        gt_field=gt,
        eval_key="eval",
        missing="fn")

    fp = sum(conf_view.values("eval_fp"))
    tp = sum(conf_view.values("eval_tp"))
    fn = sum(conf_view.values("eval_fn"))

    f1 = tp/(tp+0.5*(fp+fn))

    return -1.0*f1

def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        required=True,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/index.html#operators "
                    "for more information"
                )
            ),
        )



def get_target_view(ctx, inputs):
    has_view = ctx.view != ctx.dataset.view()
    has_selected = bool(ctx.selected)
    default_target = None

    if has_view or has_selected:
        target_choices = types.RadioGroup(orientation="horizontal")
        target_choices.add_choice(
            "DATASET",
            label="Entire dataset",
            description="Process the entire dataset",
        )

        if has_view:
            target_choices.add_choice(
                "CURRENT_VIEW",
                label="Current view",
                description="Process the current view",
            )
            default_target = "CURRENT_VIEW"

        if has_selected:
            target_choices.add_choice(
                "SELECTED_SAMPLES",
                label="Selected samples",
                description="Process only the selected samples",
            )
            default_target = "SELECTED_SAMPLES"

        inputs.enum(
            "target",
            target_choices.values(),
            default=default_target,
            required=True,
            label="Target view",
            view=target_choices,
        )

    target = ctx.params.get("target", default_target)

    return _get_target_view(ctx, target)

def _get_target_view(ctx, target):
    if target == "SELECTED_SAMPLES":
        return ctx.view.select(ctx.selected)

    if target == "DATASET":
        return ctx.dataset

    return ctx.view

def register(plugin):
    plugin.register(OptConfThresh)
