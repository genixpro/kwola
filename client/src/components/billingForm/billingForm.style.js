import styled from 'styled-components';
import TextFields from '../uielements/textfield';

const TextField = styled(TextFields)`
  margin: 0 8px 16px;
  width: calc(50% - 16px);
`;

const Form = styled.form`
  display: flex;
  flex-wrap: wrap;

  .menu {
    width: 200px;
  }
`;

export { Form, TextField };
