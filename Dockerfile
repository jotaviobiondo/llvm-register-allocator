FROM marcokuchla/ubuntu-llvm-clang


ENV USERNAME=app
ENV HOME=/home/$USERNAME

RUN useradd --user-group --create-home --shell /bin/false $USERNAME

USER $USERNAME
WORKDIR $HOME/code
