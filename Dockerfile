FROM php:8.3-apache

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

# Install system dependencies
RUN apt update && apt install -y libicu-dev \
    git unzip zlib1g-dev libpng-dev libjpeg-dev \ 
    libfreetype6-dev libwebp-dev && rm -rf /var/lib/apt/lists/*

# Configure GD
RUN docker-php-ext-configure gd --with-freetype --with-jpeg --with-webp

# Install Xdebug
RUN pecl install xdebug && \
    docker-php-ext-enable xdebug

# Install PHP extensions Type docker-php-ext-install to see available extensions
RUN docker-php-ext-install pdo_mysql intl gd
RUN echo "ServerName 127.0.0.1" >> /etc/apache2/apache2.conf

# Get latest Composer
COPY --from=composer:latest /usr/bin/composer /usr/bin/composer

# Configure virtual host
RUN mv /etc/apache2/sites-available/000-default.conf /etc/apache2/sites-available/000-default.conf.old
RUN mv /etc/apache2/sites-available/default-ssl.conf /etc/apache2/sites-available/default-ssl.conf.old
COPY /apache_conf/000-default.conf /etc/apache2/sites-available/000-default.conf
COPY /apache_conf/default-ssl.conf /etc/apache2/sites-available/default-ssl.conf

# Config files for php.ini and apache2.conf
RUN mv /usr/local/etc/php/php.ini-development /usr/local/etc/php/php.ini-development.old
RUN mv /usr/local/etc/php/php.ini-production /usr/local/etc/php/php.ini-production.old
COPY /php_conf/php.ini-development /usr/local/etc/php/
COPY /php_conf/php.ini-production /usr/local/etc/php/
COPY /apache_conf/apache2.conf /etc/apache2

# Set permissions
RUN chown -R mario:mario /var/www && \
    chmod -R 755 /var/www

# Set working directory
WORKDIR /var/www

USER ${USER_ID}

# Copy application code (with proper ownership)
COPY --chown=mario:mario . .