FROM phpmyadmin:latest

ARG TIMEZONE="Europe/Madrid"

ARG USER_ID=1000
ARG GROUP_ID=1000

# Arguments for the system user and group
ARG SYSTEM_USER="mario"
ARG SYSTEM_GROUP="mario"

# Set timezone
RUN ln -snf /usr/share/zoneinfo/${TIMEZONE} /etc/localtime && echo ${TIMEZONE} > /etc/timezone
RUN printf '[PHP]\ndate.timezone = "%s"\n', ${TIMEZONE} > /usr/local/etc/php/conf.d/tzone.ini

# Asigna grupo y usuario en contenedor para no tener que estar cambiando propietario a los archivos creados desde el contenedor
RUN groupadd --gid ${GROUP_ID} mario && \
    useradd --uid ${USER_ID} --gid ${GROUP_ID} -m mario && \
    usermod -aG www-data mario

# Set permissions
RUN chown -R mario:mario /var/www && \
    chmod -R 755 /var/www

# Set working directory
WORKDIR /var/www

USER ${USER_ID}

# Copy application code (with proper ownership)
COPY --chown=mario:mario . .