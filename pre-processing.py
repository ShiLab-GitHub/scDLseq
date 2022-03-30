import numpy as np
import pandas as pd
import os
os.environ['PATH'] += os.pathsep + os.path.expanduser('~\\AppData\\Roaming\\Python\\Lib\\site-packages\\tables')

import scanpy as sc
import operator
import codecs
import argparse

def prefilter_cells(adata,min_counts=None,max_counts=None,min_genes=200,max_genes=None):
    if min_genes is None and min_counts is None and max_genes is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[0],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,min_genes=min_genes)[0]) if min_genes is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,max_genes=max_genes)[0]) if max_genes is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_obs(id_tmp)
    adata.raw=sc.pp.log1p(adata,copy=True) 
    #check the rowname 
    print("the var_names of adata.raw: adata.raw.var_names.is_unique=:",adata.raw.var_names.is_unique)
        
def prefilter_genes(adata,min_counts=None,max_counts=None,min_cells=10,max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[1],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_cells=min_cells)[0]) if min_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_cells=max_cells)[0]) if max_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_var(id_tmp)

def prefilter_specialgenes(adata,Gene1Pattern="ERCC",Gene2Pattern="MT-"):
    id_tmp1=np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names],dtype=bool)
    id_tmp2=np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names],dtype=bool)
    id_tmp=np.logical_and(id_tmp1,id_tmp2)
    adata._inplace_subset_var(id_tmp)
    
def typeReg(cellType):
    regType = []
    for item in cellType:
        regitem = item.replace(' ','_')
        regType.append(regitem)
    return regType

def dataSeg(adata):
    sep = []
    trainNum = int(adata.n_obs*0.6)
    devNum = int(adata.n_obs*0.2)
    testNum = adata.n_obs-(trainNum+devNum)
    sep = ['train']*trainNum+['dev']*devNum+['test']*testNum
    np.random.shuffle(sep)
    return sep

def df2data(df1):
    cellType = df1.cell_type.tolist()
    df_data = df1.drop(labels='cell_type',axis=1)
    cellData = df_data.values
    return cellData,cellType

def nextGeneData(cellData):
    data = cellData
    for i in range(len(cellData)):
        currLine = cellData[i]
        yield currLine
        
def getText(cellData,cellType,geneNames,seqLength=256):
    tmpData = []
    resultGeneName = []
    for dataLine in nextGeneData(cellData):
        tmpData = [list(t) for t in zip(geneNames,dataLine.tolist())]
        tmpData.sort(key=operator.itemgetter(1),reverse=True)
        tmpData = tmpData[:seqLength]
        sortedGeneList = []
        for i in tmpData:
            sortedGeneList.append(i[0])
        resultGeneName.append(sortedGeneList)
    cellText = []
    for i in range(len(resultGeneName)):
        tmpList = []
        tmpList.append(cellType[i])
        for j in range(len(resultGeneName[i])):
            tmpList.append(resultGeneName[i][j])
        cellText.append(tmpList)
    return cellText

def writeText(target,path):
    with codecs.open(path,'a',encoding='utf-8') as f:
        for fp in target:
            for item in fp:
                f.write(item)
                f.write(' ')
            f.write('\n')
        f.close()

def main():
    parser = argparse.ArgumentParser(description='Python scRNA-seq data Pre-processing')
    
    parser.add_argument('--name', type=str, metavar='N',
                        help='The corresponding name of the experiment object')
    parser.add_argument('--annotation', type=str, metavar='A',
                        help='the original obs of annotation')
    parser.add_argument('--indir', type=str, metavar='I',
                        help='input direction')
    parser.add_argument('--outdir', type=str, metavar='I',
                        help='output direction')
    parser.add_argument('--outtype', type=str, default='all', metavar='T',
                        help='output type')
    parser.add_argument('--hvg', type=int, default=5000, metavar='H',
                        help='number of high variance genes')
    parser.add_argument('--seqlen', type=int, default=256, metavar='L',
                        help='length of marker gene sequence ')
    parser.add_argument('--ratio', type=list, default=[0.6,0.2,0.2], metavar='R',
                        help='ratio of training, validating and testing data')
    args = parser.parse_args()
    
    
    adata = sc.read(args.indir,cache=False)
    adata.var_names_make_unique(join="-")
    adata.obs_names_make_unique(join="-")
    #1.pre filter cells
    prefilter_cells(adata,min_genes=100) 
    #2 pre_filter genes
    prefilter_genes(adata,min_cells=10) # avoiding all gene is zeros
    #3 prefilter_specialgene: MT and ERCC
    prefilter_specialgenes(adata)
    #4 normalizations
    sc.pp.normalize_per_cell(adata)
    #5 HVG
    HVG = min(args.hvg,adata.n_vars)
    sc.pp.filter_genes_dispersion(adata, n_top_genes=HVG)
    #6 scale
    sc.pp.log1p(adata)
    sc.pp.scale(adata,zero_center=True)
    adata.var_names=[i.upper() for i in list(adata.var_names)]#avoding some gene have lower letter
    
    sep = dataSeg(adata)
    adata.obs['sep']=sep
    cellType = adata.obs[args.annotation].tolist()
    cellType = typeReg(cellType)
    adata.obs['celltype']=cellType
    
    if args.outtype == 'h5ad':
        for group,idx in adata.obs.groupby("sep").indices.items():
            sub_adata = adata[idx]
            sub_adata.write(args.outdir+args.name+'_'+str(group)+'.h5ad','gzip')
    elif args.outtype == 'blstm':
        for group,idx in adata.obs.groupby("sep").indices.items():
            sub_adata = adata[idx]
            cellMatrix = sub_adata.X
            cellType = sub_adata.obs['celltype'].tolist()

            geneNames = sub_adata.var_names.tolist()
            df = pd.DataFrame(data=cellMatrix, columns=geneNames)
            df['cell_type'] = cellType
            data, label = df2data(df)
            text = getText(data,label,geneNames=geneNames,seqLength=args.seqlen)
            writeText(text,args.outdir+args.name+'_'+str(group)+'.txt')
    elif args.outtype == 'scdeepsort':
        for group,idx in adata.obs.groupby("sep").indices.items():
            sub_adata = adata[idx]
            cellMatrix = sub_adata.X
            cellType = sub_adata.obs['celltype'].tolist()
            geneNames = sub_adata.var_names.tolist()
            cellMatrixT = cellMatrix.T
            nums = []
            columnNames = []
            for i in range(len(cellType)):
                columnNames.append("C_"+str(i+1))
                nums.append(i+1)
            df1 = pd.DataFrame(cellMatrixT,columns = columnNames,index = geneNames)
            df2 = pd.DataFrame(columnNames,columns = ["Cell"],index = nums)
            df2["Cell_type"] = cellType
            df1.to_csv(args.outdir+args.name+'_'+str(group)+"_data.csv")
            df2.to_csv(args.outdir+args.name+'_'+str(group)+"_label.csv")
    elif args.outtype == 'all':
        for group,idx in adata.obs.groupby("sep").indices.items():
            sub_adata = adata[idx]
            cellMatrix = sub_adata.X
            cellType = sub_adata.obs['celltype'].tolist()
            geneNames = sub_adata.var_names.tolist()
            # scdeepsort
            cellMatrixT = cellMatrix.T
            nums = []
            columnNames = []
            for i in range(len(cellType)):
                columnNames.append("C_"+str(i+1))
                nums.append(i+1)
            df1 = pd.DataFrame(cellMatrixT,columns = columnNames,index = geneNames)
            df2 = pd.DataFrame(columnNames,columns = ["Cell"],index = nums)
            df2["Cell_type"] = cellType
            df1.to_csv(args.outdir+args.name+'_'+str(group)+"_data.csv")
            df2.to_csv(args.outdir+args.name+'_'+str(group)+"_label.csv")
            ## blstm
            df = pd.DataFrame(data=cellMatrix, columns=geneNames)
            df['cell_type'] = cellType
            data, label = df2data(df)
            text = getText(data,label,geneNames=geneNames,seqLength=args.seqlen)
            writeText(text,args.outdir+args.name+'_'+str(group)+'.txt')
            ## h5ad
            sub_adata.write(args.outdir+args.name+'_'+str(group)+'.h5ad','gzip')
            
    else:
        print("please use correct keyword!")
    
if __name__ == '__main__':
    main()
