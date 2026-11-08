"""Install only named model tensors; never restore the parent's optimizer/RNG."""
from m2_u_common_v1 import require,canonical

def install_model(torch,model,checkpoint,metadata,expected_model_sha256):
    from safetensors import safe_open
    import m2_paged_checkpoint_v1 as ck
    require(canonical(metadata['model_tensors'])==expected_model_sha256,'model manifest identity')
    require(ck.aliases(model)==metadata['aliases'],'tied aliases differ before restore')
    named=ck.named_tensors(model)
    with safe_open(str(checkpoint/'model.safetensors'),framework='pt',device='cpu') as f:
        require(set(f.keys())==set(named),'model key mismatch')
        with torch.no_grad():
            for name,t in named.items():
                src=f.get_tensor(name)
                require(src.shape==t.shape and src.dtype==t.dtype,'model shape/dtype mismatch')
                t.copy_(src)
    require(ck.model_rows(torch,model)==metadata['model_tensors'] and ck.aliases(model)==metadata['aliases'],'installed model not exact')
    return model

def separate_teacher(student,teacher):
    require(student is not teacher and {p.data_ptr() for p in student.parameters()}.isdisjoint({p.data_ptr() for p in teacher.parameters()}),'student/teacher share storage')
    teacher.eval()
    for p in teacher.parameters():p.requires_grad_(False)
    require(all(p.requires_grad for p in student.parameters()) and not teacher.training,'student/teacher mode')
    return teacher
