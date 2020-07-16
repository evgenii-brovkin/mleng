# Tensorflow image
FROM tensorflow/tensorflow:2.2.0
LABEL version="0.0.5"

ARG USER="ds"
ARG UID="1023"
ARG GID="123"
ARG HOME=/home/${USER}
ARG WORKDIR=${HOME}/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

RUN addgroup --gid ${GID} dsgroup \
    && useradd -m -s /bin/bash -N -u ${UID} -g ${GID} ${USER} \
    && mkdir -p ${WORKDIR} \
    && chown ${USER}:${GID} ${WORKDIR} \
    && chmod 775 ${WORKDIR} \
    && chmod g+s ${WORKDIR}

USER ${USER}
WORKDIR ${WORKDIR}
COPY . .

CMD [ "python", "scripts/inference.py", "-d", "data", "-m", "models", "-o", "output" ]
