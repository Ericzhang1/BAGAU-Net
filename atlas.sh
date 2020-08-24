dir='/input'

cd $dir
echo "processing $dir"

if [ ! -d "t2_flair" ]; then
    mkdir t2_flair
fi

elastix -f "./pre/FLAIR.nii.gz"\
 -m "/BuckeyeAI/atlas/mni_t1.atlas.nii"\
 -p "/BuckeyeAI/atlas/affinef.txt"\
 -p "/BuckeyeAI/atlas/b_splinef.txt"\
 -out t2_flair
transformix -in "/BuckeyeAI/atlas/mni_t1.atlas.wm.mask.nii"\
 -out t2_flair\
 -tp t2_flair/TransformParameters.1.txt
cd ..