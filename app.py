import logging
from pathlib import Path

from jass.service.player_service_app import PlayerServiceApp

from agents.agent_rule_based_schieber import AgentRuleBasedSchieber
from agents.carlo import AgentCarlo
from neuronal.neuronal_agent import Neuronal
from neuronal.neuronal_rule_based_combined import NeuronalRuleBasedCombined


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
    app.add_player('carlo', AgentCarlo())
    app.add_player('neuronal_firsty', Neuronal(Path('')))
    app.add_player('neuronal_rb_combined', NeuronalRuleBasedCombined(Path('')))

    return app


if __name__ == '__main__':
   app = create_app()
   app.run()