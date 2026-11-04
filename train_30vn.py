import argparse,pathlib,json,random,csv
import torch
from torch import nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch',type=int,default=32);p.add_argument('--workers',type=int,default=2);p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    if not torch.cuda.is_available():raise SystemExit('CUDA unavailable. Check GPU installation first.')
    random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    root=pathlib.Path(a.data);out=pathlib.Path(a.output)
    weights=models.MobileNet_V3_Small_Weights.DEFAULT
    if a.evaluate:
        ckpt=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
        classes=ckpt['classes'];model=models.mobilenet_v3_small(weights=None)
    else:
        if out.exists():raise SystemExit('Output exists. Choose a NEW run directory.')
        out.mkdir(parents=True)
        model=models.mobilenet_v3_small(weights=weights)
        classes=datasets.ImageFolder(root/'train').classes
    model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,len(classes));model.cuda()
    criterion=nn.CrossEntropyLoss()
    def loader(split,train=False):
        tf=transforms.Compose([transforms.RandomResizedCrop(224,scale=(0.7,1)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]) if train else weights.transforms()
        ds=datasets.ImageFolder(root/split,transform=tf)
        if ds.classes!=classes:raise ValueError(f'Class mapping differs in {split}')
        return DataLoader(ds,batch_size=a.batch,shuffle=train,num_workers=a.workers,pin_memory=True)
    def evaluate(dl):
        model.eval();matrix=torch.zeros(len(classes),len(classes),dtype=torch.long);loss=0;total=0
        with torch.no_grad():
            for x,y in dl:
                x,y=x.cuda(),y.cuda();logits=model(x);loss+=criterion(logits,y).item()*len(y);total+=len(y)
                pred=logits.argmax(1);matrix+=torch.bincount((y.cpu()*len(classes)+pred.cpu()),minlength=len(classes)**2).reshape(len(classes),len(classes))
        tp=matrix.diag().float();support=matrix.sum(1);predicted=matrix.sum(0)
        precision=tp/predicted.clamp(min=1);recall=tp/support.clamp(min=1);f1=2*precision*recall/(precision+recall).clamp(min=1e-12)
        return {'loss':loss/total,'accuracy':tp.sum().item()/total,'macro_f1':f1.mean().item()},matrix,precision,recall,f1
    if a.evaluate:
        model.load_state_dict(ckpt['model']);metrics,cm,pr,rc,f1=evaluate(loader('test'))
        (out/'test_metrics.json').write_text(json.dumps(metrics,indent=2))
        with (out/'test_per_class.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['class','support','precision','recall','f1']);w.writerows((c,int(cm[i].sum()),float(pr[i]),float(rc[i]),float(f1[i])) for i,c in enumerate(classes))
        with (out/'test_confusion_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['actual/predicted']+classes);w.writerows([c]+cm[i].tolist() for i,c in enumerate(classes))
        print('TEST',metrics);return
    (out/'config.json').write_text(json.dumps(vars(a),indent=2));train=loader('train',True);val=loader('val')
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0003);best=-1;history=[]
    for epoch in range(a.epochs):
        model.train();running=0;total=0
        for step,(x,y) in enumerate(train,1):
            x,y=x.cuda(),y.cuda();optimizer.zero_grad(set_to_none=True);loss=criterion(model(x),y);loss.backward();optimizer.step();running+=loss.item()*len(y);total+=len(y)
            if step%100==0:print(f'Epoch {epoch+1} batch {step}/{len(train)} loss={running/total:.4f}',flush=True)
        metrics,*_=evaluate(val);row=dict(epoch=epoch+1,train_loss=running/total,**metrics);history.append(row);print(row,flush=True)
        if metrics['macro_f1']>best:
            best=metrics['macro_f1'];torch.save({'model':model.state_dict(),'classes':classes},out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2))

if __name__=='__main__':main()
