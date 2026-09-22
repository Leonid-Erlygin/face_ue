"""Source-row, metadata, and sampled native-pooling checks, with explicit limits."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from experiments.mprisk_evidence.artifacts import sha256, write_json
from experiments.evirisk_suite.protocol import required_inputs


def namekey(value):
    text=str(value).replace('\\','/').removeprefix('./')
    return text.split('/loose_crop/',1)[-1]


def correlation(a,b):
    return float(spearmanr(a,b).statistic) if len(a)>2 and np.std(a)>0 and np.std(b)>0 else None


def source_separation(frame, probe_templates, gallery_templates, dataset_name):
    """Check real source identifiers, not MS1M's documented zero placeholders.

    MXFaceDataset.create_identification_meta in
    training/dataset_classes/lightning_datasets.py writes:
        mids = np.arange(len(self.labels)); names = np.zeros_like(mids)
    Its metadata rows have four columns: name, template, media, subject.
    Media/row indices distinguish export records, NOT original image identities.
    No other dataset, sentinel value or mixed-name layout is exempted here.
    """
    names = np.asarray([namekey(x) for x in frame.iloc[:, 0]])
    templates = frame.iloc[:, 1].to_numpy()
    media = frame.iloc[:, 2].to_numpy()
    probe_rows = np.flatnonzero(np.isin(templates, probe_templates))
    gallery_rows = np.flatnonzero(np.isin(templates, gallery_templates))
    shared_templates = np.intersect1d(probe_templates, gallery_templates)
    shared_rows = np.intersect1d(probe_rows, gallery_rows)
    shared_names = np.intersect1d(names[probe_rows], names[gallery_rows])
    checks = {
        'probe_gallery_template_overlap': int(len(shared_templates)),
        'probe_gallery_metadata_row_overlap': int(len(shared_rows)),
        'probe_gallery_name_overlap': int(len(shared_names)),
        'probe_gallery_name_overlap_examples': shared_names[:10].tolist(),
        'source_identity_scope': 'metadata names; not content hashes',
    }
    warnings, blockers = [], []
    if len(shared_templates):
        blockers.append('Probe and gallery share template IDs.')
    if len(shared_rows):
        blockers.append('Probe and gallery share metadata/export row indices.')
    all_zero = bool(len(names) and np.all(names == '0'))
    is_ms1m = str(dataset_name).lower() == 'ms1m'
    if is_ms1m and all_zero:
        # Require the actual generator signature, not just a dataset-name bypass.
        generator_format = bool(frame.shape[1] == 4 and
                                np.array_equal(media, np.arange(len(frame))))
        checks['source_name_semantics'] = 'MS1M generator zero placeholders'
        checks['ms1m_generator_layout_matches'] = generator_format
        checks['source_identity_scope'] = (
            'distinct export rows/media indices; original image identity unavailable')
        checks['probe_gallery_real_source_name_overlap'] = None
        if not generator_format:
            blockers.append('MS1M zero names do not match the documented four-column '
                            'metadata with sequential media IDs; cannot interpret safely.')
        else:
            shared_media = np.intersect1d(media[probe_rows], media[gallery_rows])
            checks['probe_gallery_media_index_overlap'] = int(len(shared_media))
            if len(shared_media):
                blockers.append('MS1M probe and gallery share source media/row indices.')
            warnings.append(
                'MS1M filename 0 is a generator placeholder, not a source name. '
                'Disjoint template, metadata-row and sequential media indices are '
                'checked; original-image duplication and export order are not '
                'certified without source record IDs.')
    else:
        checks['source_name_semantics'] = 'source names'
        checks['probe_gallery_real_source_name_overlap'] = int(len(shared_names))
        if len(shared_names):
            blockers.append('Probe and gallery share source names: ' +
                            ', '.join(repr(x) for x in shared_names[:5]))
    return checks, warnings, blockers


def audit_inputs(config,data,destination,seed=777):
    root=Path(config['dataset_path']);name=str(config['dataset_name']);stem=name.lower()
    report=dict(dataset=name,checks={},warnings=[],blockers=[],source_files={})
    paths=required_inputs(config)
    namesfile=root/'meta'/f'{stem}_name_5pts_score.txt'
    if namesfile.is_file(): paths.append(namesfile)
    for p in paths: report['source_files'][str(p.resolve())]=dict(bytes=p.stat().st_size,sha256=sha256(p))
    frame=pd.read_csv(root/'meta'/f'{stem}_face_tid_mid.txt',sep=r'\s+',header=None)
    names=np.array([namekey(x) for x in frame.iloc[:,0]])
    ts=frame.iloc[:,1].to_numpy();media=frame.iloc[:,2].to_numpy()
    with np.load(root/'embeddings'/f'scf_embs_{name}.npz',allow_pickle=False) as f:
        raw=f['embs'];logk=np.asarray(f['unc'],dtype=np.float64).reshape(-1)
        idfield=next((k for k in ['image_ids','sample_ids','img_names','filenames'] if k in f),None)
        if idfield:
            try: rows=np.array([namekey(x) for x in f[idfield].reshape(-1)])
            except ValueError:
                rows=None;report['warnings'].append('Export row identifiers require pickle; deliberately not unpickled.')
            if rows is not None:
                matched=bool(np.array_equal(rows,names));report['checks']['export_row_identifiers']=dict(field=idfield,match=matched)
                if not matched: report['blockers'].append('Export row identifiers do not match template metadata order; no automatic reorder.')
        else: report['warnings'].append('No per-row identifiers in SCF export: upstream export order remains conditional on source pipeline.')
    if len(raw)!=len(names) or len(logk)!=len(raw): report['blockers'].append('Raw embeddings/uncertainty/metadata lengths disagree.')
    if namesfile.is_file():
        export=np.array([namekey(line.split()[0]) for line in namesfile.read_text().splitlines() if line.strip()])
        same=bool(np.array_equal(export,names));report['checks']['source_name_order']=same
        if not same: report['blockers'].append('Source name list and template metadata order differ.')
    else: report['warnings'].append('No source name list is available for a second row-order check.')
    gall=pd.read_csv(root/'meta'/f'{stem}_1N_gallery_G1.csv').iloc[:,0].to_numpy()
    checks, warnings, blockers = source_separation(frame, data.template_ids, gall, name)
    report['checks'].update(checks)
    report['warnings'].extend(warnings)
    report['blockers'].extend(blockers)
    if not report['blockers']:
        from evaluation.template_pooling_strategies import PoolingDefault
        rng=np.random.default_rng(seed)
        indices=np.sort(rng.choice(data.n,min(64,data.n),replace=False))
        chosen=data.template_ids[indices];keep=np.isin(ts,chosen)
        mu,kappa=PoolingDefault()(np.asarray(raw[keep],np.float64),np.exp(logk[keep])[:,None],ts[keep],media[keep])
        # Native PoolingDefault returns sorted unique template IDs.
        expected_order=np.argsort(chosen,kind='stable');expected=data.mu[indices[expected_order]]
        expected_k=data.kappa[indices[expected_order]]
        good_mu=np.allclose(mu,expected,rtol=2e-6,atol=2e-6)
        good_k=np.allclose(np.asarray(kappa).reshape(-1),expected_k,rtol=2e-6,atol=2e-6)
        report['checks']['native_pooling_sample']=dict(templates=len(indices),
            direction_max_error=float(np.max(np.abs(mu-expected))),
            concentration_max_error=float(np.max(np.abs(np.asarray(kappa).reshape(-1)-expected_k))),
            passed=bool(good_mu and good_k),full_dataset_comparison=False)
        if not good_mu or not good_k: report['blockers'].append('Sampled original pooling does not match adapter.')
        y=data.targets;by_template=dict(zip(data.template_ids,y))
        raw_y=np.array([by_template.get(t,0) for t in ts]);ix=np.flatnonzero(raw_y>0)
        cos=np.empty(len(ix),float)
        for lo in range(0,len(ix),1024):
            j=ix[lo:lo+1024];r=np.asarray(raw[j],np.float64);norm=np.linalg.norm(r,axis=1)
            if np.any(norm<=0): raise ValueError('Zero raw direction in sample-level diagnostic')
            cos[lo:lo+len(j)]=np.sum((r/norm[:,None])*data.gallery[raw_y[j]-1],axis=1)
        known=y>0;template_cos=np.sum(data.mu[known]*data.gallery[y[known]-1],axis=1)
        report['checks']['SCF_geometry']=dict(sample_n=len(ix),
            sample_spearman=correlation(logk[ix],cos),template_n=int(known.sum()),
            template_spearman=correlation(data.kappa[known],template_cos),
            definitions='All known probe images vs pooled known probe templates; true-class gallery direction. Diagnostic only.')
    report['status']='failed' if report['blockers'] else 'checked_subject_to_provenance_limits'
    report['scope']='Source metadata and sampled native pooling, not encoder retraining or proof of cross-dataset identity independence.'
    write_json(destination,report)
    if report['blockers']:
        raise ValueError(f"Input audit [{name}] at {root}: " + '; '.join(report['blockers']) +
                         f". Report: {destination}")
    return report
