#!/usr/bin/env python3
"""Independent check of L(2) values induced by saved r_k predictions.

This does not evaluate an elliptic L-function independently. Reference values
are transformations of stored numerical rational targets.
"""
import argparse
import csv
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def arctan_inverse(q, digits=100):
    with localcontext() as ctx:
        ctx.prec = digits + 20
        x = Decimal(1) / Decimal(q)
        xx = x*x
        term = x
        total = term
        n = 1
        threshold = Decimal(10) ** (-(digits+10))
        while True:
            term *= -xx
            addend = term / (2*n+1)
            total += addend
            if abs(addend) < threshold:
                break
            n += 1
        return +total


def pi_decimal(digits=100):
    with localcontext() as ctx:
        ctx.prec = digits + 10
        return +(16 * arctan_inverse(5, digits) - 4 * arctan_inverse(239, digits))


def mu_series(k, digits=100):
    """log(k) - sum_{n>=1} binom(2n,n)^2/(2n*k^(2n)).

    The ratio of successive positive terms is
    4*n*(2*n+1)^2 / ((n+1)^3*k^2), strictly below 16/k^2.
    Thus the uncomputed positive tail is at most next_term/(1-16/k^2).
    """
    assert isinstance(k, int) and k > 4
    with localcontext() as ctx:
        ctx.prec = digits + 20
        kk = Decimal(k*k)
        q = Decimal(16) / kk
        term = Decimal(2) / kk
        total = term
        n = 1
        threshold = Decimal(10) ** (-digits)
        while True:
            next_term = term * (4*n*(2*n+1)**2) / (Decimal((n+1)**3)*kk)
            tail_bound = next_term / (1-q)
            if tail_bound < threshold:
                break
            total += next_term
            term = next_term
            n += 1
        value = Decimal(k).ln() - total
        return +value, n, +tail_bound


def rational(pair):
    assert isinstance(pair, list) and len(pair) == 2
    assert all(isinstance(n, int) and not isinstance(n, bool) for n in pair)
    assert pair[1] != 0
    return Fraction(*pair)


def as_decimal(f):
    return Decimal(f.numerator)/Decimal(f.denominator)


def statistics(values):
    x = np.asarray(values, dtype=float)
    probs = [0, .001, .01, .05, .25, .5, .75, .95, .99, .999, 1]
    q = np.quantile(x, probs)
    return {
        "count": len(x),
        "mean": float(np.mean(x)),
        "sample_sd": float(np.std(x, ddof=1)),
        "quantiles": {str(p): float(t) for p,t in zip(probs,q)},
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(x))),
        "nonpositive_count": int(np.count_nonzero(x <= 0)),
        "below_0_5": int(np.count_nonzero(x < .5)),
        "above_2": int(np.count_nonzero(x > 2)),
        "above_3": int(np.count_nonzero(x > 3)),
        "above_4": int(np.count_nonzero(x > 4)),
        "above_5": int(np.count_nonzero(x > 5)),
        "above_10": int(np.count_nonzero(x > 10)),
    }


def ks_distance(x,y):
    x = np.sort(x); y = np.sort(y)
    points = np.unique(np.concatenate([x,y]))
    xcdf = np.searchsorted(x,points,side="right")/len(x)
    ycdf = np.searchsorted(y,points,side="right")/len(y)
    return float(np.max(np.abs(xcdf-ycdf)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arithmetic", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    arithmetic=json.loads(args.arithmetic.read_text())
    snapshots=json.loads(args.snapshots.read_text())
    assert arithmetic["file_label_to_paper_role"] == {"test1":"validation", "valid":"test"}
    assert snapshots["file_label_to_paper_role"] == arithmetic["file_label_to_paper_role"]
    base=arithmetic["by_file_label"]["valid"]
    assert len(base)==3000 and len({x["k"] for x in base})==3000
    selected={"1000262_18":250,"1000262_19":299,"1000262_20":300}
    expected_counts={"1000262_18":2982,"1000262_19":2979,"1000262_20":2973}
    refvalues={}
    pi=pi_decimal(100)
    max_tail_bound=Decimal(0)
    max_terms=0
    with localcontext() as ctx:
        ctx.prec=110
        for row in base:
            k,w,N=row["k"],row["w"],row["N"]
            r=rational(row["answer"])
            assert r == rational(row["stored_answer"])
            assert w in [-1,1] and N>0 and r*w>0
            mu,terms,tail=mu_series(k,100)
            Lref=Decimal(w)*4*pi*pi*mu/(Decimal(N)*as_decimal(r))
            assert mu>0 and Lref>0
            refvalues[row["example_id"]]={"mu":mu,"Lref":Lref,"r":r}
            max_tail_bound=max(max_tail_bound,tail)
            max_terms=max(max_terms,terms)
    summary={
        "input_files": {
            "arithmetic":{"path":str(args.arithmetic),"sha256":hashlib.sha256(args.arithmetic.read_bytes()).hexdigest()},
            "snapshots":{"path":str(args.snapshots),"sha256":hashlib.sha256(args.snapshots.read_bytes()).hexdigest()},
        },
        "paper_role":"test",
        "file_label":"valid",
        "reference_examples":3000,
        "prediction_replicates":3,
        "independent_test_sample_size":3000,
        "reference_construction":"omega*4*pi^2*mu(k)/(N*r_k), using the stored numerical rational candidate r_k",
        "prediction_construction":"L_ref*(r_k/s_k), with greedy predicted rational s_k",
        "not_an_independent_L_function_evaluation":True,
        "mu_series":"log(k)-sum_{n>=1} binom(2n,n)^2/(2*n*k^(2*n))",
        "mu_series_tail_bound":str(max_tail_bound),
        "mu_max_terms":max_terms,
        "decimal_working_precision":110,
        "pi_method":"Machin formula 16*atan(1/5)-4*atan(1/239), alternating series",
        "pi_check_digits":str(pi)[:102],
        "k_range":[min(r["k"] for r in base),max(r["k"] for r in base)],
        "reference_distribution":statistics([float(refvalues[r["example_id"]]["Lref"]) for r in base]),
        "runs":{},
    }
    allrows=[]
    for run, epoch in selected.items():
        data=snapshots["runs"][run]["valid"]
        assert data["paper_role"]=="test"
        assert data["selected_completed_epoch"]==epoch
        assert data["selected_epoch_index"]==epoch-1
        assert data["examples"]==len(base)==len(data["records"])
        reference=[]; predicted=[]; runrows=[]; correct=0; boundaries=0
        errors={"invalid":0,"missing":0,"zero_prediction":0,"wrong_sign":0}
        with localcontext() as ctx:
            ctx.prec=110
            for orig,row in zip(base,data["records"]):
                for key in ["example_id","k","w","N","answer","stored_answer"]:
                    assert row[key]==orig[key], (run,row["example_id"],key)
                assert not row["invalid"] and not row["missing"]
                r=rational(row["answer"])
                s=rational(row["guess_greedy"])
                assert s==rational(row["canonical_guess_greedy"])
                errors["zero_prediction"]+=int(s==0)
                errors["wrong_sign"]+=int(s*r<=0)
                assert s!=0
                factor2=(s*r>0 and abs(r)/2 < abs(s) < 2*abs(r))
                boundary=(abs(s)==abs(r)/2 or abs(s)==2*abs(r))
                assert factor2==row["factor_two_correct"]
                assert boundary==row["factor_two_boundary"]
                correct+=int(factor2);boundaries+=int(boundary)
                saved=refvalues[row["example_id"]]
                Lref=saved["Lref"]
                ratio=r/s
                Lpred=Lref*as_decimal(ratio)
                Lpred_direct=Decimal(row["w"])*4*pi*pi*saved["mu"]/(Decimal(row["N"])*as_decimal(s))
                assert abs(Lpred-Lpred_direct)/abs(Lpred) < Decimal('1e-105')
                # Equivalent multiplicative interval after reciprocal transformation.
                assert factor2==(ratio>0 and Fraction(1,2)<ratio<2)
                reference.append(float(Lref));predicted.append(float(Lpred))
                out={
                    "run_id":run,"selected_completed_epoch":epoch,"paper_role":"test",
                    "example_id":row["example_id"],"k":row["k"],"w":row["w"],"N":row["N"],
                    "target_numerator":r.numerator,"target_denominator":r.denominator,
                    "prediction_numerator":s.numerator,"prediction_denominator":s.denominator,
                    "mu_series_value":str(saved["mu"]),
                    "L_reference_from_stored_target":str(Lref),"L_from_prediction":str(Lpred),
                    "L_ratio_numerator":ratio.numerator,"L_ratio_denominator":ratio.denominator,
                    "factor_two_correct":factor2,
                    "L_relative_error":float(abs(ratio-1)),
                }
                runrows.append(out)
        assert correct==expected_counts[run]
        ref=np.array(reference);pred=np.array(predicted)
        rel=pred/ref-1
        ratio=pred/ref
        summary["runs"][run]={
            "selected_completed_epoch":epoch,
            "input_records_match_arithmetic_index_in_order":True,
            "greedy_factor_two_count":correct,
            "factor_two_accuracy_percent":100*correct/len(base),
            "strict_boundary_predictions":boundaries,
            "prediction_errors":errors,
            "predicted_distribution":statistics(pred),
            "comparison_to_reference":{
                "mean_difference":float(np.mean(pred)-np.mean(ref)),
                "sample_sd_ratio":float(np.std(pred,ddof=1)/np.std(ref,ddof=1)),
                "mean_absolute_L_error":float(np.mean(abs(pred-ref))),
                "median_absolute_L_error":float(np.median(abs(pred-ref))),
                "rmse_L":float(np.sqrt(np.mean((pred-ref)**2))),
                "mean_absolute_relative_error":float(np.mean(abs(rel))),
                "median_absolute_relative_error":float(np.median(abs(rel))),
                "relative_error_95th_percentile":float(np.quantile(abs(rel),.95)),
                "maximum_absolute_relative_error":float(np.max(abs(rel))),
                "pearson_correlation":float(np.corrcoef(ref,pred)[0,1]),
                "spearman_correlation":float(spearmanr(ref,pred).statistic),
                "distribution_wasserstein_distance":float(np.mean(abs(np.sort(ref)-np.sort(pred)))),
                "distribution_ks_distance_descriptive_only":ks_distance(ref,pred),
                "predicted_over_reference_quantiles":{str(q):float(np.quantile(ratio,q)) for q in [0,.01,.05,.25,.5,.75,.95,.99,1]},
            },
            "largest_absolute_relative_errors":[{key:row[key] for key in ["example_id","k","L_reference_from_stored_target","L_from_prediction","L_relative_error"]} for row in sorted(runrows,key=lambda r:r["L_relative_error"],reverse=True)[:20]],
            "largest_predicted_L_values":[{key:row[key] for key in ["example_id","k","L_reference_from_stored_target","L_from_prediction","L_relative_error"]} for row in sorted(runrows,key=lambda r:Decimal(r["L_from_prediction"]),reverse=True)[:10]],
        }
        allrows.extend(runrows)
    summary["warnings"]=[
        "Each run predicts the same 3000 test examples. Three replicates do not produce 9000 independent test cases.",
        "Similarity of marginal distributions alone does not establish correct per-example prediction; paired error metrics are separately recorded.",
        "The reference L-values are reconstructed from the stored numerical rational candidates. This is not new high-precision L-function verification.",
        "Epochs are the previously fixed validation selections, not selected using these test distributions.",
    ]
    (args.out/'distribution_check.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (args.out/'independent_l2_values.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0]))
        writer.writeheader();writer.writerows(allrows)
    print(json.dumps({"reference":summary["reference_distribution"],"runs":{run:{key:value for key,value in row.items() if key not in ['largest_absolute_relative_errors','largest_predicted_L_values']} for run,row in summary['runs'].items()}},indent=2))


if __name__=='__main__':
    main()
