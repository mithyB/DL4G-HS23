## Network, add a new Player
Add a new Player into `app.py`:

```
    app.add_player('rule_based', AgentRuleBasedSchieber())
```

A new player (commited and pushed to /main) will be automatically deployed to

```
    https://jass-5hhi.onrender.com/[PLAYER_NAME]
```

f.E.
```
    https://jass-5hhi.onrender.com/rule_based
```

## Docker
```
docker build --tag jassen .
```

```
docker images
```
=>

```
docker run -p 5000:5000 jassen
```