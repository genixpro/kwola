import React, { Component } from 'react';
import Input from '../../../components/uielements/input';
import IntlMessages from '../../../components/utility/intlMessages';
import LayoutWrapper from '../../../components/utility/layoutWrapper';
import PageHeader from '../../../components/utility/pageHeader';
import Papersheet, {
  DemoWrapper
} from '../../../components/utility/papersheet';
import { Row, HalfColumn } from '../../../components/utility/rowColumn';

const DefaultInputs = ({ classes }) => (
  <Papersheet title={<IntlMessages id="forms.button.defaultButtons" />}>
    <DemoWrapper>
      <Input placeholder="Hint Text" autoComplete />
      <br />
      <br />
      <Input placeholder="The hint text can be as long as you want, it will wrap." />
      <br />
      <Input id="text-field-default" defaultValue="Default Value" />
      <br />
      <Input placeholder="Hint Text" />
      <br />
      <Input defaultValue="Default Value" />
      <br />
      <Input placeholder="Hint Text" />
      <br />
      <Input placeholder="Password Field" type="password" />
      <br />
      <Input
        placeholder="MultiLine with rows: 2 and rowsMax: 4"
        multiline={true}
        rows={2}
        rowsMax={4}
      />
      <br />
      <Input placeholder="Message Field" multiline={true} rows={2} />
      <br />
      <Input placeholder="Full width" fullWidth={true} />
    </DemoWrapper>
  </Papersheet>
);

const WithErrorsInputs = ({ classes }) => (
  <Papersheet title={<IntlMessages id="forms.button.withErrorsInputs" />}>
    <DemoWrapper>
      <Input placeholder="Hint Text" error={true} />
      <br />
      <Input placeholder="Hint Text" error={true} />
      <br />
      <Input placeholder="Hint Text" error={true} />
      <br />
      <Input
        placeholder="Message Field"
        error={true}
        multiline={true}
        rows={2}
      />
      <br />
    </DemoWrapper>
  </Papersheet>
);

const InputExampleCustomize = ({ classes }) => (
  <Papersheet title={<IntlMessages id="forms.button.inputExampleCustomize" />}>
    <DemoWrapper>
      <Input placeholder="Styled Hint Text" />
      <br />
      <Input placeholder="Custom error color" error={true} />
      <br />
      <Input placeholder="Custom Underline Color" />
      <br />
      <Input placeholder="Custom Underline Focus Color" />
      <br />
      <Input />
    </DemoWrapper>
  </Papersheet>
);

const InputExampleDisabled = ({ classes }) => (
  <Papersheet title={<IntlMessages id="forms.button.inputExampleDisabled" />}>
    <DemoWrapper>
      <Input disabled={true} placeholder="Disabled Hint Text" />
      <br />
      <Input
        disabled={true}
        id="text-field-disabled"
        defaultValue="Disabled Value"
      />
      <br />
      <Input disabled={true} placeholder="Disabled Hint Text" />
      <br />
      <Input
        disabled={true}
        placeholder="Disabled Hint Text"
        defaultValue="Disabled With Floating Label"
      />
    </DemoWrapper>
  </Papersheet>
);

class InputExamples extends Component {
  render() {
    const { props } = this;
    return (
      <LayoutWrapper>
        <PageHeader>{<IntlMessages id="forms.button.header" />}</PageHeader>
        <Row>
          <HalfColumn>
            <DefaultInputs {...props} />
          </HalfColumn>
          <HalfColumn>
            <WithErrorsInputs {...props} />
          </HalfColumn>
        </Row>
        <Row>
          <HalfColumn>
            <InputExampleCustomize {...props} />
          </HalfColumn>
          <HalfColumn>
            <InputExampleDisabled {...props} />
          </HalfColumn>
        </Row>
      </LayoutWrapper>
    );
  }
}
export default InputExamples;
