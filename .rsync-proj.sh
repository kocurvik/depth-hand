#/bin/sh
PROJ_NAME=univue-hand-pose
PROJ_DIR=projects
OUT_DIR=data/univue
DATA_NAME=hands17
SERVER=${1:-palau}
MODEL=${2:-base_clean}
## upload code
SOURCE=${HOME}/${PROJ_DIR}/${PROJ_NAME}/
TARGET=${SERVER}:${PROJ_DIR}/${PROJ_NAME}/
echo uploading \
    from: [${SOURCE}] \
    to: [${TARGET}]
rsync -auvh -e ssh \
    --exclude-from='.gitignore' \
    ${SOURCE} \
    ${TARGET}
## download predictions
SOURCE=${SERVER}:${OUT_DIR}/output/${DATA_NAME}/predict/
TARGET=${HOME}/${OUT_DIR}/${SERVER}/${DATA_NAME}/predict/
mkdir -p ${TARGET}
echo downloading \
    from: [${SOURCE}] \
    to: [${TARGET}]
rsync -auvh -e ssh \
    ${SOURCE} \
    ${TARGET}
## download the full log
SOURCE=${SERVER}:${OUT_DIR}/output/${DATA_NAME}/log/blinks/${MODEL}/
TARGET=${HOME}/${OUT_DIR}/${SERVER}/${DATA_NAME}/log/${MODEL}
mkdir -p ${TARGET}
echo downloading \
    from: [${SOURCE}] \
    to: [${TARGET}]
rsync -auvh -e ssh --include='*.txt' --include='*.log' \
    ${SOURCE} \
    ${TARGET}
## download model checkpoint
SOURCE=${SERVER}:${OUT_DIR}/output/${DATA_NAME}/log/blinks/${MODEL}/model.ckpt*
TARGET=${HOME}/${OUT_DIR}/${SERVER}/${DATA_NAME}/log/${MODEL}
mkdir -p ${TARGET}
echo downloading \
    from: [${SOURCE}] \
    to: [${TARGET}]
rsync -auvh -e ssh \
    ${SOURCE} \
    ${TARGET}
