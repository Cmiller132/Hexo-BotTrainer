# Hexo Game Rules

Hexo is a two-player connection game played on a hexagonal grid.

## Goal

Be the first player to place six of your stones in a straight, connected line.

Lines can run along any of the three main axis of the grid

## Turns

- Player 0 starts by placing one stone at the center: `(0, 0)`.
- After the first move, players take turns placing two stones at a time.
- Both stones in a turn belong to the same player.

## Legal Moves

- Stones must be placed on empty hexes.
- After the opening move, each new stone must be close to the existing board.
- In the engine, "close" means within 8 hex steps of at least one stone already on the board.

## Winning

A player wins immediately after making a connected line of six stones.

There is no normal draw rule. The board is treated as unlimited.

## Simple Example

If Player 0 has stones on:

```text
(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)
```

Player 0 wins, because those six stones form one straight line.

## Threats

A threat is defined as any window of 6 tiles that has 4 or more placements from a player and 0 opponent placements
2 placements per turn allows any threat to end the game in one turn if left unblocked
This means that 4-in-a-row and 5-in-a-row are in most cases equally threatening.
In cases where there are multiple threats and blocking requires greater than 2 stones there exists a forced win. 