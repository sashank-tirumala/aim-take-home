import math
from collections import deque


def parse_grid(grid):
    """
    Parse the grid and return the positions of the thief and the policemen
    Inputs:
        grid: list of lists of integers and strings
    Outputs:
        thief_pos: tuple of two integers
        policemen_pos: dictionary of integers to tuples of two integers
    """
    if not grid or len(grid) == 1 and len(grid[0]) == 1:
        raise ValueError("Grid is empty or has only one cell")
    thief_pos = None
    policemen_pos = {}
    
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == 'T':
                if thief_pos is None:
                    thief_pos = (i, j)
                else:
                    raise ValueError("Multiple thief positions found in the grid")
            elif cell != 0:
                if cell not in policemen_pos:
                    policemen_pos[cell] = (i, j)
                else:
                    raise ValueError(f"Multiple positions found for policeman {cell} in the grid")
    
    return thief_pos, policemen_pos


def calculate_angle(p1, p2):
    dy, dx = -p2[0] + p1[0], p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360
    return angle


def is_thief_seen_by_policeman(thief_pos, policeman_pos, orientation, fov):
    """
    Check if the thief is seen by the policeman
    Inputs:
        thief_pos: tuple of two integers
        policeman_pos: tuple of two integers
        orientation: float
        fov: float
    Outputs:
        boolean
    """
    def is_point_in_fov(point):
        angle = calculate_angle(policeman_pos, point)
        angle_diff = abs(angle - orientation)
        return angle_diff < fov/2 or angle_diff > 360 - fov/2

    corners = [(thief_pos[0] - 0.5, thief_pos[1] - 0.5),
               (thief_pos[0] - 0.5, thief_pos[1] + 0.5),
               (thief_pos[0] + 0.5, thief_pos[1] - 0.5),
               (thief_pos[0] + 0.5, thief_pos[1] + 0.5)]
    
    for corner in corners:
        if is_point_in_fov(corner):
            return True

    return False




def find_closest_unseen_cell(thief_pos, policemen_pos, orientations, fovs, grid):
    """
    Find the closest unseen cell for the thief
    Inputs:
        thief_pos: tuple of two integers
        policemen_pos: dictionary of integers to tuples of two integers
        orientations: list of floats
        fovs: list of floats
        grid: list of lists of integers and strings
    Outputs:
        tuple of two integers or None
    """
    rows, cols = len(grid), len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    queue = deque([(thief_pos[0], thief_pos[1], 0)])  # (row, col, distance)
    
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while queue:
        row, col, distance = queue.popleft()
        if not visited[row][col] and grid[row][col] == 0:
            cell_pos = (row, col)
            is_unseen = True

            for cop_id, cop_pos in policemen_pos.items():
                if is_thief_seen_by_policeman(cell_pos, cop_pos, orientations[cop_id - 1], fovs[cop_id - 1]):
                    is_unseen = False
                    break

            if is_unseen:
                return cell_pos

        visited[row][col] = True

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols and not visited[new_row][new_col]:
                queue.append((new_row, new_col, distance + 1))

    return None

def main(grid, orientations, fovs):
    if not (isinstance(grid, list) and all(isinstance(row, list) for row in grid)):
        raise ValueError("Invalid grid format")

    if not (isinstance(orientations, list) and all(isinstance(o, (int, float)) for o in orientations)):
        raise ValueError("Invalid orientations format")

    if not (isinstance(fovs, list) and all(isinstance(fov, (int, float)) for fov in fovs)):
        raise ValueError("Invalid FOV format")

    thief_pos, policemen_pos = parse_grid(grid)
    seen_policemen = []

    if not policemen_pos:
        return [], thief_pos
    
    for cop_id, cop_pos in policemen_pos.items():
        if thief_pos == cop_pos:
            seen_policemen.append(cop_id)
        elif 0 <= orientations[cop_id - 1] <= 360 and 0 <= fovs[cop_id - 1] <= 360:
            if is_thief_seen_by_policeman(thief_pos, cop_pos, orientations[cop_id - 1], fovs[cop_id - 1]):
                seen_policemen.append(cop_id)
        else:
            raise ValueError(f"Invalid orientation or FOV value for policeman {cop_id}")


    closest_unseen_cell = find_closest_unseen_cell(thief_pos, policemen_pos, orientations, fovs, grid)
    return seen_policemen, closest_unseen_cell


def test_main():
    # Test case 1: Basic scenario
    grid1 = [
        [0, 0, 0, 0, 0],
        ['T', 0, 0, 0, 2],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    orientations1 = [180, 150]
    fovs1 = [60, 60]
    assert main(grid1, orientations1, fovs1) == ([2], (2, 2))

    # Test case 2: No policemen
    grid2 = [
        [0, 0, 0, 0, 0],
        ['T', 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    orientations2 = []
    fovs2 = []
    assert main(grid2, orientations2, fovs2) == ([], (1, 0))

    # Test case 3: Thief surrounded by policemen
    grid3 = [
        [1, 0, 2],
        [0, 'T', 0],
        [3, 0, 4]
    ]
    orientations3 = [270, 180, 90, 0]
    fovs3 = [180, 180, 180, 180]
    assert main(grid3, orientations3, fovs3) == ([1, 2, 3], None)

if __name__ == "__main__":

    grid = [[0, 0, 0, 0, 0], ['T', 0, 0, 0, 2], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]
    orientations = [180, 150]
    fovs = [60, 60]
    test_main()
    print(main(grid, orientations, fovs))