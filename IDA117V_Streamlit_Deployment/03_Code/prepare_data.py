"""Read original XLSX using the Python standard library; never alter sources."""
import zipfile, xml.etree.ElementTree as ET, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
ALLOWED={'A','B1','B2','B3','B4'}

def read_sheet(path,name):
    with zipfile.ZipFile(path) as z:
        ss=[''.join(t.itertext()) for t in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rel={r.attrib['Id']:r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        wb=ET.fromstring(z.read('xl/workbook.xml'))
        sheet=next(s for s in wb.find('m:sheets',NS) if s.attrib['name']==name)
        target=rel[sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']]
        target=target.lstrip('/') if target.startswith('/') else 'xl/'+target
        out=[]
        for row in ET.fromstring(z.read(target)).findall('.//m:sheetData/m:row',NS):
            d={}
            for c in row:
                v=c.find('m:v',NS)
                if v is not None:
                    value=ss[int(v.text)] if c.attrib.get('t')=='s' else v.text
                    if c.attrib.get('t') not in ['s','str','e']:
                        try:value=float(value)
                        except ValueError:pass
                    d[''.join(filter(str.isalpha,c.attrib['r']))]=value
            out.append(d)
        return out

def build():
    raw=ROOT/'02_Data/raw';add=raw/'ADDENDUM_TABLES.xlsx';maps=raw/'MAP_TABLES.xlsx'
    records=[]
    for r in read_sheet(add,'Addendum 5'):
        if r.get('D') in ALLOWED:
            records.append(dict(code=r['E'],municipality=r['F'].strip(),province=r['B'],category=r['D'],
                water_2011=r['G'],water_2022=r['H'],sanitation_2011=r['K'],sanitation_2022=r['L'],
                refuse_2011=r['M'],refuse_2022=r['N'],electricity_2011=r['O'],electricity_2022=r['P']))
    data=pd.DataFrame(records).set_index('code')
    pop={r['D']:r for r in read_sheet(add,'Addendum 1') if r.get('C') in ALLOWED}
    dwell={r['D']:r for r in read_sheet(add,'Addendum 2') if r.get('C') in ALLOWED}
    urban={r['D']:r for r in read_sheet(maps,'Map 5.2') if r.get('C') in ALLOWED}
    rdp={r['D']:r for r in read_sheet(add,'Addendum 3') if r.get('C') in ALLOWED}
    discrepancies=[]
    for code in data.index:
        p=pop[code];d=dwell[code];u=urban[code];r=rdp[code]
        for year,pc,hc,sc in [(2011,'F','G','H'),(2022,'I','J','K')]:
            data.loc[code,f'population_{year}']=p[pc];data.loc[code,f'households_{year}']=p[hc]
            data.loc[code,f'household_size_{year}']=p[sc]
        total=sum(d[k] for k in ['J','K','L','M'])
        for k,label in [('J','formal'),('K','traditional'),('L','informal'),('M','other')]:data.loc[code,f'{label}_share_2022']=100*d[k]/total
        data.loc[code,'urban_share_2022']=u['Q'];data.loc[code,'rdp_share_2022']=r['H']
        data.loc[code,'rdp_eligible_denominator_2022']=r['F']
        badsum=sum(d[k] for k in ['F','G','H','I'])
        if abs(badsum-p['G'])>max(5,p['G']*.01):
            discrepancies.append({'code':code,'dwelling_sum_2011':badsum,'households_2011':p['G'],'relative_difference':(badsum-p['G'])/p['G']})
    data['log_households_2022']=np.log1p(data.households_2022)
    data['household_growth_pct']=100*(data.households_2022/data.households_2011-1)
    data['water_backlog_pct_2022']=100-data.water_2022
    data['sanitation_backlog_pct_2022']=100-data.sanitation_2022
    data['higher_need']=((data.water_2022<90)|(data.sanitation_2022<90)).astype(int)
    data=data.reset_index().sort_values('code').reset_index(drop=True)
    assert len(data)==213 and data.code.is_unique
    assert data[['water_2022','sanitation_2022']].notna().all().all()
    data.to_csv(ROOT/'02_Data/municipal_data.csv',index=False)
    pd.DataFrame(discrepancies).to_csv(ROOT/'05_Results/source_anomalies.csv',index=False)
    audit={'records':len(data),'duplicate_codes':int(data.code.duplicated().sum()),'class_counts':data.higher_need.value_counts().to_dict(),
           'missing_values':data.isna().sum().to_dict(),'excluded_2011_dwelling_discrepancies':len(discrepancies),
           'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()},
           'invalid_percentages':{c:int((~data[c].between(0,100)).sum()) for c in data.columns if 'share' in c or c.startswith(('water_','sanitation_','refuse_','electricity_'))}}
    (ROOT/'05_Results/data_audit.json').write_text(json.dumps(audit,indent=2))
    return data

if __name__=='__main__':print(build().shape)
