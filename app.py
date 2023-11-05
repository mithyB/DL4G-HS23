import logging

from jass.service.player_service_app import PlayerServiceApp

from agents.agent_rule_based_schieber import AgentRuleBasedSchieber


def create_app():
    """
    This is the factory method for flask. It is automatically detected when flask is run, but we must tell flask
    what python file to use:

        export FLASK_APP=app.py
        export FLASK_ENV=development
        flask run --host=0.0.0.0 --port=8888
    """
    logging.basicConfig(level=logging.DEBUG)

    # create and configure the app
    app = PlayerServiceApp('player_service')

    # you could use a configuration file to load additional variables
    # app.config.from_pyfile('my_player_service.cfg', silent=False)

    # add some players
    app.add_player('rule_based', AgentRuleBasedSchieber())

    return app


if __name__ == '__main__':
   app = create_app()
   app.run()