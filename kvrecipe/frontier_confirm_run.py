from pathlib import Path
import sys
import zlib

sys.path.insert(0, str(Path(__file__).parent))
import frontier_run as fr
import controls as kc
import review_longbench as lb

WEIGHTS, RANDOM = {'Z': 0.0, 'W50': 0.5}, {'R50': 0.5}
LB_ROWS = [f'{k}.k3v2pc@{fr.B1}' for k in WEIGHTS]
LB_SPLIT, LB_EXAMPLE = lb.split_prompt, lb.run_example


def split_prompt(tokenizer, template, obj, max_length=lb.MAX_LENGTH):
    context, *rest = LB_SPLIT(tokenizer, template, obj, max_length)
    fr.STATE.update(row={'context': context, 'index': zlib.crc32(obj['_id'].encode())}, tok=tokenizer)
    return (context, *rest)


def lb_example(pipe, task, obj, prompts, max_new, capture_kvzip):
    out = LB_EXAMPLE(pipe, task, obj, prompts, max_new, capture_kvzip)
    out['frontier'] = fr.STATE['frontier']
    return out


def main():
    fr.WEIGHTS, fr.RANDOM = WEIGHTS, RANDOM if '--rows' in sys.argv else {}
    fr.configure()
    if '--longbench' in sys.argv:
        sys.argv.remove('--longbench')
        lb.ROWS, lb.split_prompt, lb.run_example, lb.hidden_scores = LB_ROWS, split_prompt, lb_example, fr.scores
        return lb.main()
    return kc.main()


if __name__ == '__main__':
    sys.exit(main())
