Assumptions:
- Models were trained on the cloud (simulate by retrieving pre-trained models and saving them locally)
- All the containers are run on the target machine.

Restrictions:
- To run on the IMG backend we dont have the option to use "--gpu all".
- Copy the GPU driver libraries to the docker: these need to be placed in a local folder ahead of time with "gather-libs.sh"

Run on the GPU host
```bash
./gather_libs.sh
docker-compose up --build
