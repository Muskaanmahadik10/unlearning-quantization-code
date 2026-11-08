"""Isolated GPTQ serialization-boundary repair; V1 evidence stays unchanged."""
import run_org_b05_behavior_cell_v1 as runner
import org_b05_packed_lifecycle_v2 as lifecycle

runner.life = lifecycle
runner.__file__ = __file__

if __name__ == '__main__':
    runner.main()
