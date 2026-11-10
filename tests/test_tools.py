import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tokenizers import Tokenizer,models,pre_tokenizers,decoders,trainers
from dialogue import encode_messages,chat_prompt,rag_user_message,RAG_NOT_FOUND
import random
import prepare_dialogue
import library
import sample_store
from verify_bundle import verify


def tokenizer():
    t=Tokenizer(models.BPE(unk_token='<unk>'))
    t.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder=decoders.ByteLevel()
    t.train_from_iterator(['Hello world.\nTitle\nElectricity runs through wires.'],
        trainers.BpeTrainer(vocab_size=300,special_tokens=['<unk>','<eos>'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return t


class ToolTests(unittest.TestCase):
    def test_assistant_mask_and_end_of_turn(self):
        tok=tokenizer()
        user=[{'role':'user','content':'Hello'}]
        messages=user+[{'role':'assistant','content':'Hello world.'}]
        prefix,_=encode_messages(tok,user)
        ids,labels=encode_messages(tok,messages)
        self.assertTrue(all(v==-100 for v in labels[:len(prefix)]))
        self.assertEqual(labels[-1],tok.token_to_id('<eos>'))
        self.assertEqual(len(ids),len(labels))
        self.assertTrue(any(v!=-100 for v in labels))

    def test_rag_examples_label_only_the_answer_and_keep_the_real_passage(self):
        tok=tokenizer()
        rows=[dict(id=str(i),title=f'T{i%3}',context=f'Passage {i} text. Electricity runs through wires.',question=f'Q{i}?',
                   answers=dict(text=['wires'] if i%2 else [],answer_start=[0] if i%2 else [])) for i in range(12)]
        made=list(prepare_dialogue.squad_conversations(rows,12,random.Random(0)))
        self.assertEqual(len(made),12)
        for messages,extra in made:
            row=rows[int(extra['squad_id'])]
            self.assertIn(row['context'],messages[0]['content'])
            self.assertTrue(messages[0]['content'].endswith('Question: '+row['question']))
            reply=messages[1]['content']
            if extra['answerable']:
                n=int(reply.split('[')[1].split(']')[0])
                self.assertIn(f'[{n}] {row["title"]}: {row["context"]}',messages[0]['content'])
            else:
                self.assertEqual(reply,RAG_NOT_FOUND)
            ids,labels=prepare_dialogue.encode(tok,messages,10_000)
            prompt,_=encode_messages(tok,messages[:1])
            self.assertTrue(all(v==-100 for v in labels[:len(prompt)]))
            self.assertEqual(tok.decode([v for v in labels if v!=-100][:-1]),reply)

    def test_over_long_conversations_are_dropped_not_cut(self):
        tok=tokenizer()
        messages=[dict(role='user',content=rag_user_message([dict(title='T',text='word '*400)],'Q?')),
                  dict(role='assistant',content='Hello world.')]
        self.assertIsNone(prepare_dialogue.encode(tok,messages,64))
        self.assertIsNotNone(prepare_dialogue.encode(tok,messages,100_000))

    def test_chat_trims_old_turns_and_preserves_latest(self):
        tok=tokenizer()
        latest=[{'role':'user','content':'Hello'}]
        expected=chat_prompt(tok,latest,100)
        long=[{'role':'user','content':'old '*100},{'role':'assistant','content':'old answer'}]+latest
        self.assertEqual(chat_prompt(tok,long,len(expected)),expected)
        with self.assertRaises(ValueError): chat_prompt(tok,latest,1)

    def test_library_resume_and_search(self):
        tok=tokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';data.mkdir();db=root/'library.sqlite'
            tok.save(str(data/'tokenizer.json'))
            sources=[dict(id='1',title='Title',url='https://en.wikipedia.org/wiki/Title'),dict(id='2',title='Hello',url='https://en.wikipedia.org/wiki/Hello')]
            packed={'train':[],'val':[]}
            for s in sources:
                split='val' if int(hashlib.sha256(s['id'].encode()).hexdigest()[:8],16)%100==0 else 'train'
                packed[split]+=tok.encode(s['title']+'\nElectricity runs through wires.').ids+[tok.token_to_id('<eos>')]
            # memmap needs nonempty files; unused validation tokens do not affect this small fixture.
            for split in packed: np.array(packed[split] or [tok.token_to_id('<eos>')],dtype='<u2').tofile(data/(split+'.bin'))
            (data/'sources.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in sources))
            (data/'metadata.json').write_text(json.dumps({'tokenizer_sha256':hashlib.sha256((data/'tokenizer.json').read_bytes()).hexdigest()}))
            library.build(data,db,1);self.assertEqual(library.status(db)['articles'],1)
            library.build(data,db,3);self.assertEqual(library.status(db)['articles'],2)
            self.assertTrue(library.status(db)['complete'])
            hits=library.search('electricity',db)
            self.assertEqual(len(hits),2)
            self.assertIn('wires',library.article(hits[0]['id'],db)['body'])
            self.assertEqual(library.search('" OR *',db),[])

    def test_samples_and_export_checksums(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            sample_store.save('full','Hi',{'step':10,'text':'Hi there'},path=root/'samples.sqlite')
            self.assertEqual(sample_store.recent(root/'samples.sqlite')[0]['result']['step'],10)
            (root/'weights').write_bytes(b'test')
            (root/'manifest.json').write_text(json.dumps({'sha256':{'weights':hashlib.sha256(b'test').hexdigest()}}))
            self.assertEqual(verify(root),1)
            (root/'weights').write_bytes(b'changed')
            with self.assertRaises(ValueError): verify(root)


if __name__=='__main__': unittest.main()
