"""Reproduce all analyses, fitting transformations on training partitions only.

Test predictions are generated once per locked candidate, after all selection,
search and ablations finish. No test score is used to change any model.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import json, platform, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest, mannwhitneyu
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.metrics import (accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,
    average_precision_score,classification_report,confusion_matrix,roc_curve,log_loss)
import joblib
from neural import TabularNeural
from prepare_data import build,ROOT

SEED=42
CAT=['province','category']
NUM=['water_2011','sanitation_2011','refuse_2011','electricity_2011',
     'log_households_2022','household_growth_pct','household_size_2022',
     'formal_share_2022','traditional_share_2022','informal_share_2022',
     'urban_share_2022','rdp_share_2022']
FEATURES=CAT+NUM
NAMES=['Dummy','Logistic regression','Random forest','MLP','Residual network']
F=ROOT/'04_Figures';R=ROOT/'05_Results'
plt.rcParams.update({'figure.dpi':140,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def savefig(name):
    plt.tight_layout();plt.savefig(F/(name+'.png'),bbox_inches='tight');plt.close()

def prep(cols):
    numeric=[c for c in NUM if c in cols];categorical=[c for c in CAT if c in cols]
    transformers=[]
    if numeric:transformers.append(('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric))
    if categorical:transformers.append(('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical))
    return ColumnTransformer(transformers)

def estimator(name,params=None):
    p=dict(params or {})
    if name=='Dummy':return DummyClassifier(strategy='most_frequent')
    if name=='Logistic regression':return LogisticRegression(max_iter=3000,class_weight=p.pop('class_weight','balanced'),random_state=SEED,**p)
    if name=='Random forest':return RandomForestClassifier(n_estimators=p.pop('n_estimators',150),max_depth=p.pop('max_depth',4),min_samples_leaf=p.pop('min_samples_leaf',5),class_weight=p.pop('class_weight','balanced'),random_state=SEED,n_jobs=1,**p)
    return TabularNeural(architecture='mlp' if name=='MLP' else 'residual',random_state=SEED,**p)

def fit(name,X,y,Xv=None,yv=None,cols=None,params=None,internal_stop=False):
    cols=cols or FEATURES;p=prep(cols);model=estimator(name,params)
    if isinstance(model,TabularNeural) and internal_stop:
        ix,iv=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=SEED)
        p.fit(X.iloc[ix][cols]);model.fit(p.transform(X.iloc[ix][cols]),np.asarray(y)[ix],eval_set=(p.transform(X.iloc[iv][cols]),np.asarray(y)[iv]))
    else:
        p.fit(X[cols]);args={}
        if isinstance(model,TabularNeural) and Xv is not None:args['eval_set']=(p.transform(Xv[cols]),yv)
        model.fit(p.transform(X[cols]),y,**args)
    return Pipeline([('prepare',p),('model',model)])

def metrics(y,prob):
    pred=prob>=.5
    return dict(accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,average='macro',zero_division=0),
      precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred,zero_division=0),
      weighted_f1=f1_score(y,pred,average='weighted',zero_division=0),roc_auc=roc_auc_score(y,prob),pr_auc=average_precision_score(y,prob))

def cv_score(name,X,y,params=None,cols=None):
    scores=[]
    for it,iv in StratifiedKFold(4,shuffle=True,random_state=SEED).split(X,y):
        m=fit(name,X.iloc[it],np.asarray(y)[it],cols=cols,params=params,internal_stop=True)
        scores.append(f1_score(np.asarray(y)[iv],m.predict(X.iloc[iv]),average='macro',zero_division=0))
    return scores

def eda(d):
    plt.figure(figsize=(6,3.5));d.higher_need.value_counts().sort_index().plot.bar(color=['#2a9d8f','#e76f51'])
    plt.xticks([0,1],['Lower access need','Higher access need'],rotation=0);plt.ylabel('Municipalities');plt.title('Approved target distribution (213 municipalities)');savefig('01_target')
    audit=json.loads((R/'data_audit.json').read_text())
    plt.figure(figsize=(7,3.4));plt.bar(['Missing predictor cells','Duplicate municipality codes','2011 dwelling anomalies'],[d[FEATURES].isna().sum().sum(),d.code.duplicated().sum(),audit['excluded_2011_dwelling_discrepancies']],color='#264653')
    plt.ylabel('Cells or municipality records');plt.title('Data quality audit: distinct issue counts');savefig('02_quality')
    for col,stem,title in [('sanitation_2011','03_historical_sanitation','Historical sanitation access and 2022 target'),('urban_share_2022','04_urban','Urban concentration and 2022 target')]:
        plt.figure(figsize=(6,3.5));plt.boxplot([d.loc[d.higher_need==v,col] for v in [0,1]],tick_labels=['Lower need','Higher need'],showfliers=True)
        plt.ylabel('Households (%)');plt.title(title);savefig(stem)
    g=d.groupby('province').higher_need.agg(['sum','count']).sort_index();g['rate']=100*g['sum']/g['count'];g.to_csv(R/'province_target_distribution.csv')
    plt.figure(figsize=(7,3.5));plt.bar(g.index,g.rate,color='#2a9d8f');plt.ylim(0,105);plt.ylabel('Municipalities with higher need (%)');plt.title('Unweighted municipal target rate by province');savefig('05_province')
    plt.figure(figsize=(6,4));plt.scatter(d.water_2022,d.sanitation_2022,c=d.higher_need,cmap='coolwarm',s=25,alpha=.75);plt.axvline(90,color='gray',ls='--');plt.axhline(90,color='gray',ls='--')
    plt.xlabel('Improved water access (%)');plt.ylabel('Improved sanitation access (%)');plt.title('Observed access and the operational 90% rule');savefig('06_access')

def main():
    started=time.time();warnings.filterwarnings('ignore',category=UserWarning)
    d=build();eda(d);y=d.higher_need.to_numpy()
    development,test=train_test_split(np.arange(len(d)),test_size=.2,stratify=y,random_state=SEED)
    train,val=train_test_split(development,test_size=.25,stratify=y[development],random_state=SEED)
    split=pd.DataFrame({'code':d.code,'partition':'train'});split.loc[val,'partition']='validation';split.loc[test,'partition']='test';split.to_csv(R/'split_manifest.csv',index=False)
    X=d[FEATURES];Xt=X.iloc[train];Xv=X.iloc[val];Xtest=X.iloc[test];yt=y[train];yv=y[val];ytest=y[test]
    models={};rows=[];cvrows=[];hist={}
    for name in NAMES:
        print('Training',name,flush=True)
        m=fit(name,Xt,yt,Xv,yv);models[name]=m
        tr=metrics(yt,m.predict_proba(Xt)[:,1]);vm=metrics(yv,m.predict_proba(Xv)[:,1])
        cv=cv_score(name,Xt,yt);cvrows.append({'model':name,'fold_scores':cv,'mean':np.mean(cv),'sd':np.std(cv,ddof=1)})
        rows.append({'model':name,'train_macro_f1':tr['macro_f1'],'train_val_gap':tr['macro_f1']-vm['macro_f1'],**{'validation_'+k:v for k,v in vm.items()},'cv_mean':np.mean(cv),'cv_sd':np.std(cv,ddof=1)})
        if name in ['MLP','Residual network']:
            h=pd.DataFrame(m.named_steps['model'].history_);hist[name]=h;h.to_csv(R/(name.replace(' ','_')+'_history.csv'),index=False)
            fig,ax=plt.subplots(1,2,figsize=(9,3.4))
            for part in ['train','val']:
                ax[0].plot(h.epoch,h[part+'_loss'],label=part);ax[1].plot(h.epoch,h[part+'_macro_f1'],label=part)
            ax[0].set(title=name+' binary cross entropy',xlabel='Epoch',ylabel='Unweighted loss');ax[1].set(title='Macro-F1 during training',xlabel='Epoch',ylabel='Macro-F1')
            for a in ax:a.legend()
            savefig(name.replace(' ','_')+'_training')
    validation=pd.DataFrame(rows);validation.to_csv(R/'validation_comparison.csv',index=False)
    ranked=validation[validation.model!='Dummy'].sort_values(['validation_macro_f1','cv_mean'],ascending=False)
    champion=ranked.iloc[0].model;comparator=ranked.iloc[1].model
    print('Validation champion:',champion,'Comparator:',comparator,flush=True)
    if champion=='Logistic regression':grid={'C':[.01,.1,1,10,100],'class_weight':['balanced',None]}
    elif champion=='Random forest':grid={'max_depth':[3,5,None],'min_samples_leaf':[2,5,10],'max_features':['sqrt',1.0]}
    else:grid={'width':[8,16],'lr':[.001,.003],'l2':[.01,.1],'dropout':[.1,.25]}
    search=[]
    for params in ParameterGrid(grid):
        scores=cv_score(champion,Xt,yt,params=params)
        search.append({'parameters':params,'cv_mean':float(np.mean(scores)),'cv_sd':float(np.std(scores,ddof=1)),'fold_scores':scores})
    best=max(search,key=lambda r:r['cv_mean']);(R/'optimisation_search.json').write_text(json.dumps(search,indent=2))
    tuned=fit(champion,Xt,yt,Xv,yv,params=best['parameters']);tunedval=metrics(yv,tuned.predict_proba(Xv)[:,1])
    baseval=metrics(yv,models[champion].predict_proba(Xv)[:,1])
    # Keep the existing champion if optimisation does not improve validation macro-F1.
    accepted=tunedval['macro_f1']>baseval['macro_f1']+1e-12
    final=tuned if accepted else models[champion];finalparams=best['parameters'] if accepted else {}
    if isinstance(final.named_steps['model'],TabularNeural):
        nn=final.named_steps['model'];h=pd.DataFrame(nn.history_)
        h.to_csv(R/'Final_champion_history.csv',index=False)
        fig,ax=plt.subplots(1,2,figsize=(9,3.4))
        for part in ['train','val']:
            ax[0].plot(h.epoch,h[part+'_loss'],label=part)
            ax[1].plot(h.epoch,h[part+'_macro_f1'],label=part)
        ax[0].set(title='Final champion loss',xlabel='Epoch',ylabel='Unweighted binary cross entropy')
        ax[1].set(title='Final champion macro-F1',xlabel='Epoch',ylabel='Macro-F1')
        for a in ax:a.legend()
        savefig('Final_champion_training')
        trscore=metrics(yt,final.predict_proba(Xt)[:,1])['macro_f1'];vscore=metrics(yv,final.predict_proba(Xv)[:,1])['macro_f1']
        details={'architecture':nn.get_params(),'epochs_run':nn.n_iter_,'best_epoch':nn.best_epoch_,'train_macro_f1':trscore,'validation_macro_f1':vscore,'parameter_count':sum(v.size for v in nn.params.values()),'train_validation_gap':trscore-vscore}
        (R/'final_training_details.json').write_text(json.dumps(details,indent=2))
    ablations=[]
    groups={'Remove historical service indicators':['water_2011','sanitation_2011','refuse_2011','electricity_2011'],
            'Remove geography':['province','category'],
            'Remove housing and urban context':['formal_share_2022','traditional_share_2022','informal_share_2022','urban_share_2022','rdp_share_2022']}
    finalvm=metrics(yv,final.predict_proba(Xv)[:,1])
    for label,excluded in groups.items():
        cols=[c for c in FEATURES if c not in excluded];a=fit(champion,Xt,yt,Xv,yv,cols=cols,params=finalparams)
        score=metrics(yv,a.predict_proba(Xv)[:,1])['macro_f1']
        ablations.append({'experiment':label,'validation_macro_f1':score,'delta':score-finalvm['macro_f1'],'removed_features':', '.join(excluded)})
    pd.DataFrame(ablations).to_csv(R/'ablations.csv',index=False)
    plt.figure(figsize=(8,3.5));plt.barh([r['experiment'] for r in ablations],[r['delta'] for r in ablations],color='#e76f51');plt.axvline(0,color='gray');plt.xlabel('Change in validation macro-F1');plt.title('Controlled feature-group ablations');savefig('ablations')
    # Learning curves use the same fixed validation set and seed, never the test set.
    lc=[]
    for name in NAMES[1:]:
        for fraction in [.35,.55,.75,1.0]:
            ix=np.arange(len(yt)) if fraction==1 else train_test_split(np.arange(len(yt)),train_size=fraction,stratify=yt,random_state=SEED)[0]
            m=fit(name,Xt.iloc[ix],yt[ix],Xv,yv)
            lc.append({'model':name,'training_n':len(ix),'train_macro_f1':metrics(yt[ix],m.predict_proba(Xt.iloc[ix])[:,1])['macro_f1'],'validation_macro_f1':metrics(yv,m.predict_proba(Xv)[:,1])['macro_f1']})
    pd.DataFrame(lc).to_csv(R/'learning_curves.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(9,6))
    for name,ax in zip(NAMES[1:],axs.ravel()):
        q=pd.DataFrame(lc);q=q[q.model==name];ax.plot(q.training_n,q.train_macro_f1,label='Train');ax.plot(q.training_n,q.validation_macro_f1,label='Validation');ax.set(title=name,xlabel='Training municipalities',ylabel='Macro-F1',ylim=(0,1));ax.legend()
    savefig('learning_curves')
    # LOCK POINT. All candidate choices and experiments are complete above this line.
    allmodels={**models,'Final champion':final};testrows=[];predictions=d.iloc[test][['code','municipality','province','category']].copy();predictions['actual']=ytest
    fig,ax=plt.subplots(figsize=(6.5,4.5))
    for name,m in allmodels.items():
        prob=m.predict_proba(Xtest)[:,1];pred=(prob>=.5).astype(int);testrows.append({'model':name,**metrics(ytest,prob)})
        predictions[name+'_probability']=prob;predictions[name+'_prediction']=pred
        report=classification_report(ytest,pred,labels=[0,1],target_names=['Lower need','Higher need'],output_dict=True,zero_division=0)
        (R/(name.replace(' ','_')+'_classification_report.json')).write_text(json.dumps(report,indent=2))
        cm=confusion_matrix(ytest,pred,labels=[0,1]);pd.DataFrame(cm,index=['Actual lower','Actual higher'],columns=['Predicted lower','Predicted higher']).to_csv(R/(name.replace(' ','_')+'_confusion_matrix.csv'))
        fig2,a=plt.subplots(figsize=(4,3));a.imshow(cm,cmap='Blues');a.set(xticks=[0,1],yticks=[0,1],xticklabels=['Lower','Higher'],yticklabels=['Lower','Higher'],xlabel='Predicted class',ylabel='Actual class',title=name+' test confusion matrix')
        for i in range(2):
            for j in range(2):a.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
        fig2.tight_layout();fig2.savefig(F/(name.replace(' ','_')+'_confusion.png'),bbox_inches='tight');plt.close(fig2)
        fpr,tpr,_=roc_curve(ytest,prob);ax.plot(fpr,tpr,label=f'{name} ({roc_auc_score(ytest,prob):.2f})')
    ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Locked models on the same test municipalities');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(F/'roc_curves.png',bbox_inches='tight');plt.close(fig)
    testframe=pd.DataFrame(testrows);testframe.to_csv(R/'test_metrics.csv',index=False);predictions.to_csv(R/'paired_test_predictions.csv',index=False)
    plt.figure(figsize=(8,3.8));plt.bar(testframe.model,testframe.macro_f1,color='#2a9d8f');plt.xticks(rotation=20,ha='right');plt.ylim(0,1);plt.ylabel('Test macro-F1');plt.title('Final model comparison');savefig('model_comparison')
    a=predictions['Final champion_prediction'].to_numpy()==ytest;b=predictions[comparator+'_prediction'].to_numpy()==ytest
    both=int((a&b).sum());onlya=int((a&~b).sum());onlyb=int((~a&b).sum());neither=int((~a&~b).sum())
    exact=float(binomtest(onlya,onlya+onlyb,.5).pvalue) if onlya+onlyb else 1.
    mc={'both_correct':both,'champion_only_correct':onlya,'comparator_only_correct':onlyb,'both_wrong':neither,'discordant':onlya+onlyb,'exact_p':exact,'alpha':.05,'comparator':comparator,'statistic_continuity_corrected':max(abs(onlya-onlyb)-1,0)**2/(onlya+onlyb) if onlya+onlyb else 0,'independence_caveat':'Municipal spatial dependence may violate independent-pair assumptions; inference is exploratory.'}
    (R/'mcnemar.json').write_text(json.dumps(mc,indent=2))
    # Percentile paired bootstrap intervals; exploratory under municipal dependence.
    rng=np.random.default_rng(SEED);ci=[]
    for name in ['Final champion',comparator]:
        yp=predictions[name+'_prediction'].to_numpy();values=[]
        for _ in range(2000):
            ix=rng.integers(0,len(ytest),len(ytest));values.append(f1_score(ytest[ix],yp[ix],average='macro',labels=[0,1],zero_division=0))
        ci.append({'model':name,'macro_f1':f1_score(ytest,yp,average='macro'),'low':float(np.quantile(values,.025)),'high':float(np.quantile(values,.975))})
    pd.DataFrame(ci).to_csv(R/'bootstrap_intervals.csv',index=False)
    fairness=[]
    for grouping in ['province','category']:
        for group,q in predictions.groupby(grouping):
            yy=q.actual.to_numpy();pp=q['Final champion_prediction'].to_numpy();fairness.append({'grouping':grouping,'group':group,'n':len(q),'higher_need_support':int(yy.sum()),'accuracy':accuracy_score(yy,pp),'macro_f1':f1_score(yy,pp,average='macro',labels=[0,1],zero_division=0),'higher_need_recall':recall_score(yy,pp,zero_division=0) if yy.sum() else None,'warning':'Small subgroup: descriptive only'})
    pd.DataFrame(fairness).to_csv(R/'subgroup_metrics.csv',index=False)
    # Target-threshold sensitivity changes labels only; no retuning or repeated test scoring.
    pd.DataFrame([{'access_threshold_pct':t,'higher_need':int(((d.water_2022<t)|(d.sanitation_2022<t)).sum()),'lower_need':int(((d.water_2022>=t)&(d.sanitation_2022>=t)).sum())} for t in [85,90,95]]).to_csv(R/'target_sensitivity.csv',index=False)
    descriptive=d[NUM+['water_2022','sanitation_2022']].describe().T;descriptive.to_csv(R/'descriptive_statistics.csv')
    # Development data only for exploratory tests of feature differences.
    tests=[]
    for c in ['sanitation_2011','urban_share_2022']:
        q=d.iloc[development];u,p=mannwhitneyu(q.loc[q.higher_need==0,c],q.loc[q.higher_need==1,c],alternative='two-sided')
        tests.append({'feature':c,'U':float(u),'p_value':float(p),'interpretation':'Exploratory association, no causal claim; municipal independence assumption may fail.'})
    pd.DataFrame(tests).to_csv(R/'exploratory_tests.csv',index=False)
    joblib.dump({'pipeline':final,'features':FEATURES,'champion':champion,'target_threshold':90,'prediction_threshold':.5,'training_codes':d.iloc[train].code.tolist()},ROOT/'03_Code/final_model.joblib')
    example=d.iloc[test[:3]][FEATURES].copy();example.to_csv(ROOT/'02_Data/demo_valid_inputs.csv',index=False)
    invalid=example.iloc[:1].copy();invalid['water_2011']=150;invalid.to_csv(ROOT/'02_Data/demo_invalid_input.csv',index=False)
    summary={'seed':SEED,'train_n':len(train),'validation_n':len(val),'test_n':len(test),'champion':champion,'comparator':comparator,'search_space':grid,'best_cv_configuration':best,'optimisation_accepted':bool(accepted),'final_parameters':finalparams,'pre_optimisation_validation':baseval,'searched_validation':tunedval,'final_validation':finalvm,'final_test':testframe[testframe.model=='Final champion'].iloc[0].to_dict(),'mcnemar':mc,'runtime_seconds':time.time()-started,'environment':{'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn_version},'feature_columns':FEATURES}
    (R/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
