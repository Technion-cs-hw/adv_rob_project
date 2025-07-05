# %%
from  projectCode.imports import *

# %%

device = set_device()
ann_file = "./annotations/annotations/captions_val2017.json"
num_examples = 256
batch_size = 1
seed = 42


# %%
ds = CocoDataset(ann_file, num_examples, img_dir = "imgs",seed = seed)
dl = DataLoader(ds, batch_size = batch_size)


# %%

for blur_kernel_size in [3,5,7,9,11]:
    tokens_path = './models/ExpansionNet_v2/demo_material/demo_coco_tokens.pickle'
    model_params={"device":device,"tokens_path":tokens_path,"blur_kernel_size":blur_kernel_size,"blured":True}
    model = loadModel("ExpansionNet",model_path="./models/ExpansionNet_v2/rf_model.pth",**model_params)
    
    
    attack_type = "untargeted"
    target = "door"
    token_id = getExpNetTokenId(target,tokens_path)
    print(token_id)

    # %%
    for bound in [8]:#,16,32]:
        print("epsilon bound:",bound)
        eps = bound/255
        iter_per_attack = 50 #25
         
        base = "./adversarial_attacks"+"/ExpNet"+f"blured{blur_kernel_size}"+f"/eps{bound}"+f"/it{iter_per_attack}_exNum{num_examples}"
        attacks_file_name = base + ".pt" 
        captions_file = base+".json"

        # %%
        # GENERATING ADVERSARIAL EXAMPLES
        
        attacker = Attacker(model,torch.nn.MSELoss(),device, num_iterations=iter_per_attack, epsilon=eps)
        if attack_type == "targeted":
            pert_fn = lambda x: attacker.perturbExpNetTargeted(x,token_id)
        elif attack_type == "untargeted":
            pert_fn = attacker.perturbExpNet

        print(attacks_file_name)
        if not os.path.exists(attacks_file_name):
            attacks = []

            for batch in tqdm(dl):
                pert = pert_fn(batch[0])
                attacks.append(pert)
            perts = torch.cat(attacks)
            torch.save(perts,attacks_file_name)
        else:
            perts = torch.load(attacks_file_name)


        # %%
        captions = None
        caps = None

        #use if exists already
        if  os.path.exists(captions_file):
            with open(captions_file, 'r') as openfile:
                captions = json.load(openfile)
                if "captions" in captions.keys():
                    caps = captions["captions"]
            
        if caps is None:
            caps = []
            for i in range(num_examples):
                x = ds[i]
                pert = perts[i]
                
                out, _, _, _ = model(x[0].unsqueeze(0).to(device)) # prediction, pred_probs, logits, cross_enc_output
                
                p_out, _, _, _ = model(torch.clamp((x[0].to(device)+pert.to(device)).unsqueeze(0),0,1))
                caps.append((out,p_out))

            if captions is None:
                captions =  {
                "seed": seed,
                "num_examples": num_examples,
                "batch_size": batch_size,
                "captions": caps
                }
            else:
                captions["captions"] = caps

            with open(captions_file, 'w') as openfile:
                    json.dump(captions,openfile)

        # %%
        #use if already exists
        if "scores" in captions.keys():
            meteor_scores = captions["scores"]["meteor"]

        else:
            meteor_scores=[]
            for capt,ex in zip(caps,ds):
                c_clear = capt[0]
                c_corr = capt[1]
                img = ex[0]

                
                score =  meteor_eval_single(ex[1],c_clear[0])
                score_corr = meteor_eval_single(ex[1],c_corr[0])
                
                
                meteor_scores.append((score,score_corr))#,out,p_out)) #img, delta, model_output, model_output_corrupted

            captions["scores"]={}
            captions["scores"]["meteor"] = meteor_scores

            with open(captions_file, 'w') as openfile:
                json.dump(captions,openfile)

   


