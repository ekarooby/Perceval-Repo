import perceval as pcvl

MY_TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjU5MiwiZXhwIjoxODA1OTM3MjE2Ljk5ODA0NTd9.4CcQuD0_Oo9Kh7ptB-7pNFCNoM1cCy1smMicszHN3De9X88Dd05J6l89GkveCIiWwqG7H3QHI_Sq7Vk91F53Vw"
QPU_NAME = "qpu:belenos"

pcvl.RemoteConfig.set_token(MY_TOKEN)
pcvl.RemoteConfig().save()

JOB_ID = "6224b2ac-7369-46d2-839d-151389aaabaf"

remote_proc = pcvl.RemoteProcessor(QPU_NAME)
job = remote_proc.resume_job(JOB_ID)
results = job.get_results()

circuit = results.get('computed_circuit')

print(f"Number of components : {circuit.ncomponents()}")

# Count component types
n_bs = 0
n_ps = 0
n_other = 0
for _, c in circuit._components:
    name = type(c).__name__
    if 'BS' in name:
        n_bs += 1
    elif 'PS' in name:
        n_ps += 1
    else:
        n_other += 1

print(f"Beam splitters (BS)  : {n_bs}")
print(f"Phase shifters (PS)  : {n_ps}")
print(f"Other components     : {n_other}")
print(f"Total modes          : {circuit.m}")