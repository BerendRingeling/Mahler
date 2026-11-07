#!/usr/bin/env python3
"""Compare the main L2 CSV outputs against independent high-precision values."""
import argparse
import csv
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--main-dir',type=Path,required=True)
    ap.add_argument('--independent-dir',type=Path,required=True)
    args=ap.parse_args()
    main=args.main_dir
    indy=args.independent_dir
    allrows=read_csv(indy/'independent_l2_values.csv')
    own_summary=json.loads((indy/'distribution_check.json').read_text())
    primary_summary=json.loads((main/'distribution_summary.json').read_text())
    by_key={(r['run_id'],r['example_id']):r for r in allrows}
    assert len(allrows)==len(by_key)==9000
    run_ids=['1000262_18','1000262_19','1000262_20']
    own_refs=[row for row in allrows if row['run_id']==run_ids[0]]
    ref_by_id={r['example_id']:r for r in own_refs}
    assert len(ref_by_id)==len(own_refs)==3000
    checks=0
    def exact(x,y,context):
        nonlocal checks
        checks+=1
        assert x==y,(context,x,y)
    def close(x,y,context,tol=1e-12):
        nonlocal checks
        checks+=1
        x=float(x);y=float(y)
        assert math.isfinite(x) and math.isfinite(y),context
        assert math.isclose(x,y,rel_tol=tol,abs_tol=tol),(context,x,y)

    errors={name:{'max_absolute':0.0,'max_relative':0.0} for name in ['mu','reference_L2','induced_L2','induced_over_reference','absolute_relative_error']}
    def numerical(main_value,independent_value,name,context):
        close(main_value,independent_value,context)
        with localcontext() as ctx:
            ctx.prec=110
            a=Decimal(main_value)
            b=Decimal(independent_value)
            ab=abs(a-b)
            rel=ab/abs(b) if b else ab
        errors[name]['max_absolute']=max(errors[name]['max_absolute'],float(ab))
        errors[name]['max_relative']=max(errors[name]['max_relative'],float(rel))

    ref_rows=read_csv(main/'l2_reference_values.csv')
    exact(len(ref_rows),3000,'reference length')
    exact([r['example_id'] for r in ref_rows],[r['example_id'] for r in own_refs],'reference order')
    for row in ref_rows:
        own=ref_by_id[row['example_id']]
        for key in ['example_id','k','w','N','target_numerator','target_denominator']:
            exact(row[key],own[key],('reference metadata',row['example_id'],key))
        numerical(row['mu'],own['mu_series_value'],'mu',('mu',row['example_id']))
        numerical(row['reference_L2'],own['L_reference_from_stored_target'],'reference_L2',('Lref',row['example_id']))

    pred_rows=read_csv(main/'l2_predictions.csv')
    exact(len(pred_rows),9000,'prediction length')
    exact([(r['run_id'],r['example_id']) for r in pred_rows],list(by_key),'prediction order')
    own_counts={run:0 for run in run_ids}
    for row in pred_rows:
        own=by_key[row['run_id'],row['example_id']]
        for main_key, own_key in [('run_id','run_id'),('selected_completed_epoch','selected_completed_epoch'),('example_id','example_id'),('k','k'),('guess_numerator','prediction_numerator'),('guess_denominator','prediction_denominator')]:
            exact(row[main_key],own[own_key],('prediction metadata',row['run_id'],row['example_id'],main_key))
        numerical(row['reference_L2'],own['L_reference_from_stored_target'],'reference_L2',('predLref',row['run_id'],row['example_id']))
        numerical(row['induced_L2'],own['L_from_prediction'],'induced_L2',('inducedL',row['run_id'],row['example_id']))
        with localcontext() as ctx:
            ctx.prec=110
            ratio=Decimal(own['L_ratio_numerator'])/Decimal(own['L_ratio_denominator'])
            error=abs(ratio-1)
        numerical(row['induced_over_reference'],str(ratio),'induced_over_reference',('ratio',row['run_id'],row['example_id']))
        numerical(row['absolute_relative_error'],str(error),'absolute_relative_error',('error',row['run_id'],row['example_id']))
        factor2=int(own['factor_two_correct']=='True')
        exact(int(row['factor_two_correct']),factor2,('factor2',row['run_id'],row['example_id']))
        own_counts[row['run_id']]+=factor2

    # Compute bin indices from the high-precision Decimal values and exact
    # decimal-width bins, independently of NumPy and the primary float arrays.
    # Read only the intended bin definition from the main summary, rather than
    # hardcoding a particular display or deriving bins from rounded CSV edges.
    histogram_policy=primary_summary['histogram']
    bin_width=Decimal(str(histogram_policy['bin_width']))
    bin_count=int(histogram_policy['number_of_bins'])
    lower,upper=[Decimal(str(value)) for value in histogram_policy['full_range']]
    assert bin_width>0 and bin_count>0 and upper>lower
    exact(bin_width*bin_count,upper-lower,'declared bin range and width')
    sample_count=len(own_refs)
    density_normalizer=Decimal(sample_count)*bin_width
    bins={'reference':[0]*bin_count,**{run:[0]*bin_count for run in run_ids}}
    nearest_distance=upper-lower
    def histogram_bin(value):
        nonlocal nearest_distance
        with localcontext() as ctx:
            ctx.prec=110
            value=Decimal(value)
            assert lower<=value<=upper
            scaled=(value-lower)/bin_width
            i=int(scaled)
            nearest_distance=min(nearest_distance,(scaled-i)*bin_width,(i+1-scaled)*bin_width)
            if i==bin_count:i=bin_count-1
            return i
    for row in own_refs:
        bins['reference'][histogram_bin(row['L_reference_from_stored_target'])]+=1
    for row in allrows:
        bins[row['run_id']][histogram_bin(row['L_from_prediction'])]+=1
    hist=read_csv(main/'l2_histogram_counts.csv')
    exact(len(hist),bin_count,'histogram bins')
    for i,row in enumerate(hist):
        close(row['bin_left'],lower+i*bin_width,('left edge',i))
        close(row['bin_right'],lower+(i+1)*bin_width,('right edge',i))
        exact(int(row['reference_count']),bins['reference'][i],('reference bin count',i))
        close(row['reference_density'],Decimal(bins['reference'][i])/density_normalizer,('reference density',i))
        model_counts=[]
        for run in run_ids:
            exact(int(row[run+'_count']),bins[run][i],('prediction bin count',run,i))
            model_counts.append(bins[run][i])
        avg=statistics.mean(model_counts)
        sd=statistics.stdev(model_counts)
        close(row['model_mean_count'],avg,('model mean count',i))
        close(row['model_sample_sd_count'],sd,('model sd count',i))
        close(row['model_mean_density'],avg/float(density_normalizer),('model mean density',i))
        close(row['model_sample_sd_density'],sd/float(density_normalizer),('model sd density',i))
    for name,count_list in bins.items():
        exact(sum(count_list),3000,('total histogram count',name))

    # Summary comparisons, including all tail counts relevant to cropping.
    own_ref_values=[Decimal(row['L_reference_from_stored_target']) for row in own_refs]
    minimum=min(own_ref_values);maximum=max(own_ref_values)
    exact(primary_summary['test_k_range'],own_summary['k_range'],'k range')
    for name, section in [('reference',primary_summary['reference'])]+[(run,primary_summary['runs'][run]) for run in run_ids]:
        expected=own_summary['reference_distribution'] if name=='reference' else own_summary['runs'][name]['predicted_distribution']
        exact(section['examples'],3000,('summary examples',name))
        for key in ['below_0_5','above_2']:
            exact(section[key],expected[key],('summary count',name,key))
        for key in ['below_0','zero','nonfinite']:
            exact(section[key],0,('summary zero',name,key))
        for key in ['mean','sample_sd']:
            close(section[key],expected[key],('summary metric',name,key))
        for key,quantile in [('minimum','0'),('maximum','1'),('median','0.5')]:
            close(section[key],expected['quantiles'][quantile],('summary quantile',name,key))
        for quantile,value in section['quantiles_numpy_linear'].items():
            close(value,expected['quantiles'][quantile],('summary quantile',name,quantile))
        if name!='reference':
            own=own_summary['runs'][name]
            close(section['median_absolute_relative_error'],own['comparison_to_reference']['median_absolute_relative_error'],('median relative error',name))
            close(section['p95_absolute_relative_error'],own['comparison_to_reference']['relative_error_95th_percentile'],('p95 relative error',name))
            close(section['mean_absolute_error'],own['comparison_to_reference']['mean_absolute_L_error'],('mean absolute error',name))
            exact(section['selection']['completed_epoch'],own['selected_completed_epoch'],('selection epoch',name))
            exact(section['selection']['epoch_index'],own['selected_completed_epoch']-1,('selection index',name))
            exact(section['selection']['test_factor_two_count_from_original_audit'],own_counts[name],('original factor2 count',name))
            exact(section['test_factor_two_count_recomputed'],own_counts[name],('recomputed factor2 count',name))
            close(section['test_factor_two_accuracy_percent'],100*own_counts[name]/3000,('factor2 percent',name))
            values=[Decimal(row['L_from_prediction']) for row in allrows if row['run_id']==name]
            exact(section['below_reference_minimum'],sum(x<minimum for x in values),('below reference range',name))
            exact(section['above_reference_maximum'],sum(x>maximum for x in values),('above reference range',name))
            close(section['histogram_mass_at_or_below_2'],sum(x<=2 for x in values)/3000,('central mass',name))

    result={
        'all_checks_pass':True,
        'checks':checks,
        'comparison_tolerance_relative_and_absolute':1e-12,
        'reference_rows_compared':3000,
        'prediction_rows_compared':9000,
        'histogram_bins_per_distribution':bin_count,
        'histogram_bin_width':str(bin_width),
        'histogram_full_range':[str(lower),str(upper)],
        'exact_histogram_count_comparisons':bin_count*(1+len(run_ids)),
        'histogram_counts_all_3000':True,
        'nearest_value_distance_to_exact_bin_edge':str(nearest_distance),
        'numerical_errors_main_versus_independent':errors,
        'main_summary_agrees':True,
        'selected_epochs_and_factor_two_counts_agree':True,
        'input_hashes':{name:hashlib.sha256((main/name).read_bytes()).hexdigest() for name in ['l2_reference_values.csv','l2_predictions.csv','l2_histogram_counts.csv','distribution_summary.json']},
    }
    (indy/'main_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
