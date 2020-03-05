import React from 'react';
import { Form } from 'formik';
import Button from '@material-ui/core/Button';
import TextField from '@material-ui/core/TextField';
import HeadingWithIcon from '../HeadingWithIcon/HeadingWithIcon';
import FolderIcon from './05-icon.svg';
import RenderColumnWrapper from './RenderColumnForm.style';

const RenderColumnForm = props => {
  const {
    handleSubmit,
    onCancel,
    values,
    touched,
    errors,
    handleChange,
  } = props;
  return (
    <RenderColumnWrapper className="render-form-wrapper">
      {!props.initials && (
        <HeadingWithIcon
          iconSrc={FolderIcon}
          heading={'Column Name'}
          size={'20px'}
        />
      )}
      <Form onSubmit={handleSubmit}>
        <TextField
          name="title"
          helperText={touched.title ? errors.title : ''}
          error={Boolean(errors.title)}
          label="Column Name"
          value={values.title}
          onChange={handleChange}
          fullWidth
        />
        <Button
          type="submit"
          style={{ marginRight: 15 }}
          variant="contained"
          color="primary"
        >
          {props.initials && props.initials.editing ? 'Update' : 'Create'}
        </Button>
        <Button
          type="button"
          onClick={onCancel}
          variant="contained"
          color="secondary"
        >
          Cancel
        </Button>
      </Form>
    </RenderColumnWrapper>
  );
};

export default RenderColumnForm;
